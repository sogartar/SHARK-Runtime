// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/pending_queue_actions.h"

#include <stdbool.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/event_pool.h"
#include "iree/hal/drivers/hip/event_semaphore.h"
#include "iree/hal/drivers/hip/graph_command_buffer.h"
#include "iree/hal/drivers/hip/hip_device.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/drivers/hip/stream_command_buffer.h"
#include "iree/hal/drivers/utils/semaphore.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/resource_set.h"

// The maximal number of hipEvent_t objects a command buffer can wait.
#define IREE_HAL_HIP_MAX_WAIT_EVENT_COUNT 32

//===----------------------------------------------------------------------===//
// Queue action
//===----------------------------------------------------------------------===//

typedef enum iree_hal_hip_queue_action_kind_e {
  IREE_HAL_HIP_QUEUE_ACTION_TYPE_EXECUTION,
  // TODO: Add support for queue alloca and dealloca.
} iree_hal_hip_queue_action_kind_t;

typedef enum iree_hal_hip_queue_action_state_e {
  // The current action is active as waiting for or under execution.
  IREE_HAL_HIP_QUEUE_ACTION_STATE_ALIVE,
  // The current action is done execution and waiting for destruction.
  IREE_HAL_HIP_QUEUE_ACTION_STATE_ZOMBIE,
} iree_hal_hip_queue_action_state_t;

// A pending queue action.
//
// Note that this struct does not have internal synchronization; it's expected
// to work together with the pending action queue, which synchronizes accesses.
typedef struct iree_hal_hip_queue_action_t {
  // Intrusive doubly-linked list next entry pointer.
  struct iree_hal_hip_queue_action_t* next;
  // Intrusive doubly-linked list previous entry pointer.
  struct iree_hal_hip_queue_action_t* prev;

  // The owning pending actions queue. We use its allocators and pools.
  // Retained to make sure it outlives the current action.
  iree_hal_hip_pending_queue_actions_t* owning_actions;

  // The current state of this action. When an action is initially created it
  // will be alive and enqueued to wait for releasing to the GPU. After done
  // execution, it will be flipped into zombie state and enqueued again for
  // destruction.
  iree_hal_hip_queue_action_state_t state;
  // The callback to run after completing this action and before freeing
  // all resources. Can be NULL.
  iree_hal_hip_pending_action_cleanup_callback_t cleanup_callback;
  // User data to pass into the callback.
  void* callback_user_data;

  iree_hal_hip_queue_action_kind_t kind;
  union {
    struct {
      iree_host_size_t count;
      iree_hal_command_buffer_t** ptr;
    } command_buffers;
  } payload;

  // The device from which to allocate HIP stream-based command buffers for
  // applying deferred command buffers.
  iree_hal_device_t* device;

  // The stream to launch main GPU workload.
  hipStream_t dispatch_hip_stream;
  // The stream to launch HIP host function callbacks.
  hipStream_t callback_hip_stream;

  // Resource set to retain all associated resources by the payload.
  iree_hal_resource_set_t* resource_set;

  // Semaphore list to wait on for the payload to start on the GPU.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphore list to signal after the payload completes on the GPU.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Scratch fields for analyzing whether actions are ready to issue.
  iree_hal_hip_event_t* events[IREE_HAL_HIP_MAX_WAIT_EVENT_COUNT];
  iree_host_size_t event_count;
  // Whether the current action is still not ready for releasing to the GPU.
  bool is_pending;
} iree_hal_hip_queue_action_t;

static void iree_hal_hip_queue_action_clear_events(
    iree_hal_hip_queue_action_t* action) {
  for (iree_host_size_t i = 0; i < action->event_count; ++i) {
    iree_hal_hip_event_release(action->events[i]);
  }
  action->event_count = 0;
}

static void iree_hal_hip_queue_action_destroy(
    iree_hal_hip_queue_action_t* action);

//===----------------------------------------------------------------------===//
// Queue action list
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hip_queue_action_list_t {
  iree_hal_hip_queue_action_t* head;
  iree_hal_hip_queue_action_t* tail;
} iree_hal_hip_queue_action_list_t;

// Returns true if the action list is empty.
static inline bool iree_hal_hip_queue_action_list_is_empty(
    const iree_hal_hip_queue_action_list_t* list) {
  return list->head == NULL;
}

// Pushes |action| on to the end of the given action |list|.
static void iree_hal_hip_queue_action_list_push_back(
    iree_hal_hip_queue_action_list_t* list,
    iree_hal_hip_queue_action_t* action) {
  if (list->tail) {
    list->tail->next = action;
  } else {
    list->head = action;
  }
  action->next = NULL;
  action->prev = list->tail;
  list->tail = action;
}

// Erases |action| from |list|.
static void iree_hal_hip_queue_action_list_erase(
    iree_hal_hip_queue_action_list_t* list,
    iree_hal_hip_queue_action_t* action) {
  iree_hal_hip_queue_action_t* next = action->next;
  iree_hal_hip_queue_action_t* prev = action->prev;
  if (prev) {
    prev->next = next;
    action->prev = NULL;
  } else {
    list->head = next;
  }
  if (next) {
    next->prev = prev;
    action->next = NULL;
  } else {
    list->tail = prev;
  }
}

// Takes all actions from |available_list| and moves them into |ready_list|.
static void iree_hal_hip_queue_action_list_take_all(
    iree_hal_hip_queue_action_list_t* available_list,
    iree_hal_hip_queue_action_list_t* ready_list) {
  IREE_ASSERT_NE(available_list, ready_list);
  ready_list->head = available_list->head;
  ready_list->tail = available_list->tail;
  available_list->head = NULL;
  available_list->tail = NULL;
}

static void iree_hal_hip_queue_action_list_destroy(
    iree_hal_hip_queue_action_t* head_action) {
  while (head_action) {
    iree_hal_hip_queue_action_t* next_action = head_action->next;
    iree_hal_hip_queue_action_destroy(head_action);
    head_action = next_action;
  }
}

//===----------------------------------------------------------------------===//
// Ready-list processing
//===----------------------------------------------------------------------===//

// Ready action atomic slist entry struct.
typedef struct iree_hal_hip_atomic_slist_entry_t {
  iree_hal_hip_queue_action_t* ready_list_head;
  iree_atomic_slist_intrusive_ptr_t slist_next;
} iree_hal_hip_atomic_slist_entry_t;

// Ready action atomic slist.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_hal_hip_ready_action,
                                iree_hal_hip_atomic_slist_entry_t,
                                offsetof(iree_hal_hip_atomic_slist_entry_t,
                                         slist_next));

static void iree_hal_hip_ready_action_slist_destroy(
    iree_hal_hip_ready_action_slist_t* list, iree_allocator_t host_allocator) {
  while (true) {
    iree_hal_hip_atomic_slist_entry_t* entry =
        iree_hal_hip_ready_action_slist_pop(list);
    if (!entry) break;
    iree_hal_hip_queue_action_list_destroy(entry->ready_list_head);
    iree_allocator_free(host_allocator, entry);
  }
  iree_hal_hip_ready_action_slist_deinitialize(list);
}

static iree_hal_hip_queue_action_t* iree_hal_hip_atomic_slist_entry_pop_front(
    iree_hal_hip_atomic_slist_entry_t* list) {
  IREE_ASSERT(list->ready_list_head);

  iree_hal_hip_queue_action_t* action = list->ready_list_head;
  if (action->next) action->next->prev = NULL;
  action->next = NULL;
  list->ready_list_head = action->next;

  return action;
}

// The ready-list processing worker's working/exiting state.
//
// States in the list has increasing priorities--meaning normally ones appearing
// earlier can overwrite ones appearing later without checking; but not the
// reverse order.
typedef enum iree_hal_hip_worker_state_e {
  IREE_HAL_HIP_WORKER_STATE_IDLE_WAITING = 0,      // Worker to main thread
  IREE_HAL_HIP_WORKER_STATE_WORKLOAD_PENDING = 1,  // Main to worker thread
  IREE_HAL_HIP_WORKER_STATE_EXIT_REQUESTED = -1,   // Main to worker thread
  IREE_HAL_HIP_WORKER_STATE_EXIT_COMMITTED = -2,   // Worker to main thread
  IREE_HAL_HIP_WORKER_STATE_EXIT_ERROR = -3,       // Worker to main thread
} iree_hal_hip_worker_state_t;

// The data structure needed by a ready-list processing worker thread to issue
// ready actions to the GPU.
//
// This data structure is shared between the parent thread, which owns the
// whole pending actions queue, and the worker thread; so proper synchronization
// is needed to touch it from both sides.
//
// The parent thread should push a list of ready actions to ready_worklist,
// update worker_state, and give state_notification accordingly.
// The worker thread waits on the state_notification and checks worker_state,
// and pops from the ready_worklist to process. The worker thread also monintors
// worker_state and stops processing if requested by the parent thread.
typedef struct iree_hal_hip_working_area_t {
  // Notification from the parent thread to request worker state changes.
  iree_notification_t state_notification;
  // Notification to the parent thread to indicate the worker committed exiting.
  // TODO: maybe remove this. We can just wait on the worker thread to exit.
  iree_notification_t exit_notification;
  iree_hal_hip_ready_action_slist_t ready_worklist;  // atomic
  iree_atomic_int32_t worker_state;                  // atomic
  // TODO: use status to provide more context for the error.
  iree_atomic_intptr_t error_code;                   // atomic
  // The number of actions that have been issued to the GPU but not yet fully
  // completed both execution and cleanup. We don't need this field to be atomic
  // given it is modified only from the worker thread.
  int32_t pending_action_count;

  // The number of HIP stream callbacks that are currently scheduled for
  // execution on the hot stream.
  // We need to wait for them to finish before destrying the context.
  iree_slim_mutex_t host_stream_pending_callbacks_count_mutex;
  iree_notification_t host_stream_pending_callbacks_count_notification;
  int32_t host_stream_pending_callbacks_count
      IREE_GUARDED_BY(host_stream_pending_callbacks_count_mutex);

  iree_allocator_t host_allocator;  // const
} iree_hal_hip_working_area_t;

static void iree_hal_hip_working_area_initialize(
    iree_allocator_t host_allocator,
    iree_hal_hip_working_area_t* working_area) {
  iree_notification_initialize(&working_area->state_notification);
  iree_notification_initialize(&working_area->exit_notification);
  iree_hal_hip_ready_action_slist_initialize(&working_area->ready_worklist);
  iree_atomic_store_int32(&working_area->worker_state,
                          IREE_HAL_HIP_WORKER_STATE_IDLE_WAITING,
                          iree_memory_order_release);
  iree_atomic_store_int32(&working_area->error_code, IREE_STATUS_OK,
                          iree_memory_order_release);
  working_area->pending_action_count = 0;
  iree_slim_mutex_initialize(
      &working_area->host_stream_pending_callbacks_count_mutex);
  iree_notification_initialize(
      &working_area->host_stream_pending_callbacks_count_notification);
  working_area->host_stream_pending_callbacks_count = 0;
  working_area->host_allocator = host_allocator;
}

static void iree_hal_hip_working_area_deinitialize(
    iree_hal_hip_working_area_t* working_area) {
  iree_hal_hip_ready_action_slist_destroy(&working_area->ready_worklist,
                                          working_area->host_allocator);
  iree_notification_deinitialize(&working_area->exit_notification);
  iree_notification_deinitialize(&working_area->state_notification);
  iree_slim_mutex_deinitialize(
      &working_area->host_stream_pending_callbacks_count_mutex);
  iree_notification_deinitialize(
      &working_area->host_stream_pending_callbacks_count_notification);
}

// The main function for the ready-list processing worker thread.
static int iree_hal_hip_worker_execute(
    iree_hal_hip_working_area_t* working_area);

//===----------------------------------------------------------------------===//
// Pending queue actions
//===----------------------------------------------------------------------===//

struct iree_hal_hip_pending_queue_actions_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  // The allocator used to create the timepoint pool.
  iree_allocator_t host_allocator;
  // The block pool to allocate resource sets from.
  iree_arena_block_pool_t* block_pool;

  // The symbols used to create and destroy hipEvent_t objects.
  const iree_hal_hip_dynamic_symbols_t* symbols;

  // Non-recursive mutex guarding access to the action list.
  iree_slim_mutex_t action_mutex;

  // The double-linked list of pending actions.
  iree_hal_hip_queue_action_list_t action_list IREE_GUARDED_BY(action_mutex);

  // The worker thread that monitors incoming requests and issues ready actions
  // to the GPU.
  iree_thread_t* worker_thread;
  // The worker's working area; data exchange place with the parent thread.
  iree_hal_hip_working_area_t working_area;
};

static const iree_hal_resource_vtable_t
    iree_hal_hip_pending_queue_actions_vtable;

iree_status_t iree_hal_hip_pending_queue_actions_create(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_hip_pending_queue_actions_t** out_actions) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_actions);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_pending_queue_actions_t* actions = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*actions),
                                (void**)&actions));
  iree_hal_resource_initialize(&iree_hal_hip_pending_queue_actions_vtable,
                               &actions->resource);
  actions->host_allocator = host_allocator;
  actions->block_pool = block_pool;
  actions->symbols = symbols;
  iree_slim_mutex_initialize(&actions->action_mutex);
  memset(&actions->action_list, 0, sizeof(actions->action_list));

  // Initialize the working area for the ready-list processing worker.
  iree_hal_hip_working_area_t* working_area = &actions->working_area;
  iree_hal_hip_working_area_initialize(host_allocator, working_area);

  // Create the ready-list processing worker itself.
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = IREE_SV("deferque_worker");
  params.create_suspended = false;
  iree_status_t status = iree_thread_create(
      (iree_thread_entry_t)iree_hal_hip_worker_execute, working_area, params,
      actions->host_allocator, &actions->worker_thread);

  if (iree_status_is_ok(status)) {
    *out_actions = actions;
  } else {
    iree_hal_hip_pending_queue_actions_destroy((iree_hal_resource_t*)actions);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_hal_hip_pending_queue_actions_t*
iree_hal_hip_pending_queue_actions_cast(iree_hal_resource_t* base_value) {
  return (iree_hal_hip_pending_queue_actions_t*)base_value;
}

static bool iree_hal_hip_worker_committed_exiting(
    iree_hal_hip_working_area_t* working_area);

void iree_hal_hip_pending_queue_actions_destroy(
    iree_hal_resource_t* base_actions) {
  iree_hal_hip_pending_queue_actions_t* actions =
      iree_hal_hip_pending_queue_actions_cast(base_actions);
  iree_allocator_t host_allocator = actions->host_allocator;
  iree_hal_hip_working_area_t* working_area = &actions->working_area;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Request the worker to exit.
  iree_hal_hip_worker_state_t prev_state =
      (iree_hal_hip_worker_state_t)iree_atomic_exchange_int32(
          &working_area->worker_state, IREE_HAL_HIP_WORKER_STATE_EXIT_REQUESTED,
          iree_memory_order_acq_rel);
  iree_notification_post(&working_area->state_notification, IREE_ALL_WAITERS);

  // Check potential exit states from the worker.
  if (prev_state != IREE_HAL_HIP_WORKER_STATE_EXIT_ERROR) {
    // Wait until the worker acknowledged exiting.
    iree_notification_await(
        &working_area->exit_notification,
        (iree_condition_fn_t)iree_hal_hip_worker_committed_exiting,
        working_area, iree_infinite_timeout());
  }

  // Now we can delete worker related resources.
  iree_thread_release(actions->worker_thread);
  iree_hal_hip_working_area_deinitialize(working_area);

  iree_slim_mutex_deinitialize(&actions->action_mutex);
  iree_hal_hip_queue_action_list_destroy(actions->action_list.head);
  iree_allocator_free(host_allocator, actions);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_resource_vtable_t
    iree_hal_hip_pending_queue_actions_vtable = {
        .destroy = iree_hal_hip_pending_queue_actions_destroy,
};

// Copies of the given |in_list| to |out_list| to retain the command buffer
// list.
static iree_status_t iree_hal_hip_copy_command_buffer_list(
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* in_list, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t*** out_list) {
  *out_list = NULL;
  if (!command_buffer_count) return iree_ok_status();

  iree_host_size_t total_size = command_buffer_count * sizeof(*in_list);
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)out_list));
  memcpy((void*)*out_list, in_list, total_size);
  return iree_ok_status();
}

// Frees the semaphore and value list inside |semaphore_list|.
static void iree_hal_hip_free_command_buffer_list(
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t* const* command_buffer_list) {
  iree_allocator_free(host_allocator, (void*)command_buffer_list);
}

// Copies of the given |in_list| to |out_list| to retain the semaphore and value
// list.
static iree_status_t iree_hal_hip_copy_semaphore_list(
    iree_hal_semaphore_list_t in_list, iree_allocator_t host_allocator,
    iree_hal_semaphore_list_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  if (!in_list.count) return iree_ok_status();

  out_list->count = in_list.count;
  iree_host_size_t semaphore_size = in_list.count * sizeof(*in_list.semaphores);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, semaphore_size,
                                             (void**)&out_list->semaphores));
  memcpy(out_list->semaphores, in_list.semaphores, semaphore_size);

  iree_host_size_t value_size = in_list.count * sizeof(*in_list.payload_values);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, value_size, (void**)&out_list->payload_values));
  memcpy(out_list->payload_values, in_list.payload_values, value_size);
  return iree_ok_status();
}

// Frees the semaphore and value list inside |semaphore_list|.
static void iree_hal_hip_free_semaphore_list(
    iree_allocator_t host_allocator,
    iree_hal_semaphore_list_t* semaphore_list) {
  iree_allocator_free(host_allocator, semaphore_list->semaphores);
  iree_allocator_free(host_allocator, semaphore_list->payload_values);
}

static void iree_hal_hip_queue_action_destroy(
    iree_hal_hip_queue_action_t* action) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_hip_pending_queue_actions_t* actions = action->owning_actions;
  iree_allocator_t host_allocator = actions->host_allocator;

  // Call user provided callback before releasing any resource.
  if (action->cleanup_callback) {
    action->cleanup_callback(action->callback_user_data);
  }

  // Only release resources after callbacks have been issued.
  iree_hal_resource_set_free(action->resource_set);
  iree_hal_hip_free_semaphore_list(host_allocator,
                                   &action->wait_semaphore_list);
  iree_hal_hip_free_semaphore_list(host_allocator,
                                   &action->signal_semaphore_list);

  iree_hal_hip_queue_action_clear_events(action);

  iree_hal_resource_release(actions);

  iree_allocator_free(host_allocator, action);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hip_pending_queue_actions_enqueue_execution(
    iree_hal_device_t* device, hipStream_t dispatch_stream,
    hipStream_t callback_stream, iree_hal_hip_pending_queue_actions_t* actions,
    iree_hal_hip_pending_action_cleanup_callback_t cleanup_callback,
    void* callback_user_data,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  IREE_ASSERT_ARGUMENT(actions);
  IREE_ASSERT_ARGUMENT(command_buffer_count == 0 || command_buffers);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_queue_action_t* action = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(actions->host_allocator, sizeof(*action),
                                (void**)&action));

  action->owning_actions = actions;
  action->state = IREE_HAL_HIP_QUEUE_ACTION_STATE_ALIVE;
  action->cleanup_callback = cleanup_callback;
  action->callback_user_data = callback_user_data;
  action->kind = IREE_HAL_HIP_QUEUE_ACTION_TYPE_EXECUTION;
  action->device = device;
  action->dispatch_hip_stream = dispatch_stream;
  action->callback_hip_stream = callback_stream;

  // Initialize scratch fields.
  action->event_count = 0;
  action->is_pending = true;

  // Retain all command buffers and semaphores.
  iree_hal_resource_set_t* resource_set = NULL;
  iree_status_t status =
      iree_hal_resource_set_allocate(actions->block_pool, &resource_set);
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status = iree_hal_resource_set_insert(resource_set, command_buffer_count,
                                          command_buffers);
  }
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status =
        iree_hal_resource_set_insert(resource_set, wait_semaphore_list.count,
                                     wait_semaphore_list.semaphores);
  }
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status =
        iree_hal_resource_set_insert(resource_set, signal_semaphore_list.count,
                                     signal_semaphore_list.semaphores);
  }
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    action->resource_set = resource_set;
  }

  // Copy the command buffer list for later access.
  // TODO: avoid host allocator malloc; use some pool for the allocation.
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    action->payload.command_buffers.count = command_buffer_count;
    status = iree_hal_hip_copy_command_buffer_list(
        command_buffer_count, command_buffers, actions->host_allocator,
        &action->payload.command_buffers.ptr);
  }

  // Copy the semaphore and value list for later access.
  // TODO: avoid host allocator malloc; use some pool for the allocation.
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status = iree_hal_hip_copy_semaphore_list(wait_semaphore_list,
                                              actions->host_allocator,
                                              &action->wait_semaphore_list);
  }
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status = iree_hal_hip_copy_semaphore_list(signal_semaphore_list,
                                              actions->host_allocator,
                                              &action->signal_semaphore_list);
  }

  if (IREE_LIKELY(iree_status_is_ok(status))) {
    // Retain the owning queue to make sure the action outlives it.
    iree_hal_resource_retain(actions);

    // Now everything is okay and we can enqueue the action.
    iree_slim_mutex_lock(&actions->action_mutex);
    iree_hal_hip_queue_action_list_push_back(&actions->action_list, action);
    iree_slim_mutex_unlock(&actions->action_mutex);
  } else {
    iree_hal_hip_free_semaphore_list(actions->host_allocator,
                                     &action->wait_semaphore_list);
    iree_hal_hip_free_semaphore_list(actions->host_allocator,
                                     &action->signal_semaphore_list);
    iree_hal_hip_free_command_buffer_list(actions->host_allocator,
                                          action->payload.command_buffers.ptr);
    iree_hal_resource_set_free(resource_set);
    iree_allocator_free(actions->host_allocator, action);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hip_post_error_to_worker_state(
    iree_hal_hip_working_area_t* working_area, iree_status_code_t code) {
  // Write error code, but don't overwrite existing error codes.
  intptr_t prev_error_code = IREE_STATUS_OK;
  iree_atomic_compare_exchange_strong_int32(
      &working_area->error_code, /*expected=*/&prev_error_code,
      /*desired=*/code,
      /*order_succ=*/iree_memory_order_acq_rel,
      /*order_fail=*/iree_memory_order_acquire);

  // This state has the highest priority so just overwrite.
  iree_atomic_store_int32(&working_area->worker_state,
                          IREE_HAL_HIP_WORKER_STATE_EXIT_ERROR,
                          iree_memory_order_release);
  iree_notification_post(&working_area->state_notification, IREE_ALL_WAITERS);
}

// Releases resources after action completion on the GPU and advances timeline
// and pending actions queue.
//
// This is the HIP host function callback to hipLaunchHostFunc(), invoked by a
// HIP driver thread. Note that code in this function MUST NOT invoke any GPU
// API under the hood to avoid potential deadlock.
static void iree_hal_hip_execution_device_signal_host_callback(
    void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_hip_queue_action_t* action = (iree_hal_hip_queue_action_t*)user_data;
  IREE_ASSERT_EQ(action->kind, IREE_HAL_HIP_QUEUE_ACTION_TYPE_EXECUTION);
  IREE_ASSERT_EQ(action->state, IREE_HAL_HIP_QUEUE_ACTION_STATE_ALIVE);
  iree_hal_hip_pending_queue_actions_t* actions = action->owning_actions;

  iree_status_t status;

  // Need to signal the list before zombifying the action, because in the mean
  // time someone else may issue the pending queue actions.
  // If we push first to the pending actions list, the cleanup of this action
  // may run while we are still using the semaphore list, causing a crash.
  status = iree_hal_semaphore_list_signal(action->signal_semaphore_list);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    IREE_ASSERT(false && "cannot signal semaphores in host callback");
    iree_hal_hip_post_error_to_worker_state(&actions->working_area,
                                            iree_status_code(status));
  }

  // Flip the action state to zombie and enqueue it again so that we can let
  // the worker thread clean it up. Note that this is necessary because cleanup
  // may involve GPU API calls like buffer releasing or unregistering, so we can
  // not inline it here.
  action->state = IREE_HAL_HIP_QUEUE_ACTION_STATE_ZOMBIE;
  iree_slim_mutex_lock(&actions->action_mutex);
  iree_hal_hip_queue_action_list_push_back(&actions->action_list, action);
  iree_slim_mutex_unlock(&actions->action_mutex);

  // We need to trigger execution of this action again, so it gets cleaned up.
  status = iree_hal_hip_pending_queue_actions_issue(actions);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    IREE_ASSERT(false && "cannot issue action for cleanup in host callback");
    iree_hal_hip_post_error_to_worker_state(&actions->working_area,
                                            iree_status_code(status));
  }

  iree_slim_mutex_lock(
      &actions->working_area.host_stream_pending_callbacks_count_mutex);
  --actions->working_area.host_stream_pending_callbacks_count;
  if (actions->working_area.host_stream_pending_callbacks_count == 0) {
    // Notify inside the lock to make sure that we are done touching anything
    // since the context may get destroyed in the meantime.
    iree_notification_post(
        &actions->working_area.host_stream_pending_callbacks_count_notification,
        IREE_ALL_WAITERS);
  }
  iree_slim_mutex_unlock(
      &actions->working_area.host_stream_pending_callbacks_count_mutex);

  IREE_TRACE_ZONE_END(z0);
}

// Issues the given kernel dispatch |action| to the GPU.
static iree_status_t iree_hal_hip_pending_queue_actions_issue_execution(
    iree_hal_hip_queue_action_t* action) {
  IREE_ASSERT_EQ(action->kind, IREE_HAL_HIP_QUEUE_ACTION_TYPE_EXECUTION);
  IREE_ASSERT_EQ(action->is_pending, false);
  const iree_hal_hip_dynamic_symbols_t* symbols =
      action->owning_actions->symbols;
  IREE_TRACE_ZONE_BEGIN(z0);

  // No need to lock given that this action is already detched from the pending
  // actions list; so only this thread is seeing it now.

  // First wait all the device hipEvent_t in the dispatch stream.
  for (iree_host_size_t i = 0; i < action->event_count; ++i) {
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, symbols,
        hipStreamWaitEvent(action->dispatch_hip_stream,
                           iree_hal_hip_event_handle(action->events[i]),
                           /*flags=*/0),
        "hipStreamWaitEvent");
  }

  // Then launch all command buffers to the dispatch stream.
  IREE_TRACE_ZONE_BEGIN(dispatch_command_buffers);
  IREE_TRACE_ZONE_APPEND_TEXT(dispatch_command_buffers,
                              " dispatch_command_buffers",
                              strlen(" dispatch_command_buffers"));
  for (iree_host_size_t i = 0; i < action->payload.command_buffers.count; ++i) {
    iree_hal_command_buffer_t* command_buffer =
        action->payload.command_buffers.ptr[i];
    if (iree_hal_hip_stream_command_buffer_isa(command_buffer)) {
      // Nothing to do for an inline command buffer; all the work has already
      // been submitted. When we support semaphores we'll still need to signal
      // their completion but do not have to worry about any waits: if there
      // were waits we wouldn't have been able to execute inline!
    } else if (iree_hal_hip_graph_command_buffer_isa(command_buffer)) {
      hipGraphExec_t exec = iree_hal_hip_graph_command_buffer_handle(
          action->payload.command_buffers.ptr[i]);
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, symbols, hipGraphLaunch(exec, action->dispatch_hip_stream),
          "hipGraphLaunch");
    } else {
      iree_hal_command_buffer_t* stream_command_buffer = NULL;
      iree_hal_command_buffer_mode_t mode =
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_hip_device_create_stream_command_buffer(
                  action->device, mode, IREE_HAL_COMMAND_CATEGORY_ANY,
                  /*binding_capacity=*/0, &stream_command_buffer));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(action->resource_set, 1,
                                           &stream_command_buffer));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_deferred_command_buffer_apply(
                  command_buffer, stream_command_buffer,
                  iree_hal_buffer_binding_table_empty()));
    }
  }
  IREE_TRACE_ZONE_END(dispatch_command_buffers);

  // Last record hipEvent_t signals in the dispatch stream.
  for (iree_host_size_t i = 0; i < action->signal_semaphore_list.count; ++i) {
    // Grab a hipEvent_t for this semaphore value signaling.
    hipEvent_t event = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hip_event_semaphore_acquire_timepoint_device_signal(
                action->signal_semaphore_list.semaphores[i],
                action->signal_semaphore_list.payload_values[i], &event));

    // Record the event signaling in the dispatch stream.
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, symbols, hipEventRecord(event, action->dispatch_hip_stream),
        "hipEventRecord");
    // Let the callback stream to wait on the hipEvent_t.
    IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
        z0, symbols,
        hipStreamWaitEvent(action->callback_hip_stream, event, /*flags=*/0),
        "hipStreamWaitEvent");
  }

  // Increase the pending action counter. We decrease it once it fully
  // completes and gets cleaned up.
  ++action->owning_actions->working_area.pending_action_count;

  iree_slim_mutex_lock(&action->owning_actions->working_area
                            .host_stream_pending_callbacks_count_mutex);
  ++action->owning_actions->working_area.host_stream_pending_callbacks_count;
  iree_slim_mutex_unlock(&action->owning_actions->working_area
                              .host_stream_pending_callbacks_count_mutex);

  // Now launch a host function on the callback stream to advance the semaphore
  // timeline.
  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, symbols,
      hipLaunchHostFunc(action->callback_hip_stream,
                        iree_hal_hip_execution_device_signal_host_callback,
                        action),
      "hipLaunchHostFunc");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Performs the given cleanup |action| on the CPU.
static void iree_hal_hip_pending_queue_actions_issue_cleanup(
    iree_hal_hip_queue_action_t* action) {
  iree_hal_hip_pending_queue_actions_t* actions = action->owning_actions;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_queue_action_destroy(action);

  // Now we fully executed and cleaned up this action. Decrease the pending
  // action counter.
  --actions->working_area.pending_action_count;

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hip_pending_queue_actions_issue(
    iree_hal_hip_pending_queue_actions_t* actions) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_queue_action_list_t pending_list = {NULL, NULL};
  iree_hal_hip_queue_action_list_t ready_list = {NULL, NULL};

  iree_slim_mutex_lock(&actions->action_mutex);

  if (iree_hal_hip_queue_action_list_is_empty(&actions->action_list)) {
    iree_slim_mutex_unlock(&actions->action_mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Scan through the list and categorize actions into pending and ready lists.
  iree_status_t status = iree_ok_status();
  iree_hal_hip_queue_action_t* action = actions->action_list.head;
  while (action) {
    iree_hal_hip_queue_action_t* next_action = action->next;
    action->next = NULL;

    iree_hal_semaphore_t** semaphores = action->wait_semaphore_list.semaphores;
    uint64_t* values = action->wait_semaphore_list.payload_values;

    action->is_pending = false;

    // Cleanup actions are immediately ready to release. Otherwise, look at all
    // wait semaphores to make sure that they are either already ready or we can
    // wait on a device event.
    if (action->state == IREE_HAL_HIP_QUEUE_ACTION_STATE_ALIVE) {
      for (iree_host_size_t i = 0; i < action->wait_semaphore_list.count; ++i) {
        // If this semaphore has already signaled past the desired value, we can
        // just ignore it.
        uint64_t value = 0;
        status = iree_hal_semaphore_query(semaphores[i], &value);
        if (IREE_UNLIKELY(!iree_status_is_ok(status))) break;
        if (value >= values[i]) {
          // No need to wait on this timepoint as it has already occurred and
          // we can remove it from the wait list.
          iree_hal_semaphore_list_remove_element(&action->wait_semaphore_list,
                                                 i);
          --i;
          continue;
        }

        // Try to acquire a HIP event from an existing device signal timepoint.
        // If so, we can use that event to wait on the device.
        // Otherwise, this action is still not ready for execution.
        // Before issuing recording on a stream, an event represents an empty
        // set of work so waiting on it will just return success.
        // Here we must guarantee the HIP event is indeed recorded, which means
        // it's associated with some already present device signal timepoint on
        // the semaphore timeline.
        iree_hal_hip_event_t* wait_event = NULL;
        if (!iree_hal_hip_semaphore_acquire_event_host_wait(
                semaphores[i], values[i], &wait_event)) {
          action->is_pending = true;
          break;
        }
        if (IREE_UNLIKELY(action->event_count >=
                          IREE_HAL_HIP_MAX_WAIT_EVENT_COUNT)) {
          status = iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "exceeded maximum queue action wait event limit");
          iree_hal_hip_event_release(wait_event);
          break;
        }
        action->events[action->event_count++] = wait_event;

        // Remove the wait timepoint as we have a corresponding event that we
        // will wait on.
        iree_hal_semaphore_list_remove_element(&action->wait_semaphore_list, i);
        --i;
      }
    }

    if (IREE_UNLIKELY(!iree_status_is_ok(status))) break;

    if (action->is_pending) {
      iree_hal_hip_queue_action_list_push_back(&pending_list, action);
    } else {
      iree_hal_hip_queue_action_list_push_back(&ready_list, action);
    }

    action = next_action;
  }

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    // Some error happened during processing the current action.
    // Put it back to the pending list so we don't leak.
    action->is_pending = true;
    iree_hal_hip_queue_action_list_push_back(&pending_list, action);
  }

  // Preserve pending timepoints.
  actions->action_list = pending_list;

  iree_slim_mutex_unlock(&actions->action_mutex);

  if (ready_list.head == NULL) {
    // Nothing ready yet. Just return.
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_hip_atomic_slist_entry_t* entry = NULL;
  // TODO: avoid host allocator malloc; use some pool for the allocation.
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(actions->host_allocator, sizeof(*entry),
                                   (void**)&entry);
  }

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    // Release all actions in the ready list to avoid leaking.
    iree_hal_hip_queue_action_list_destroy(ready_list.head);
    iree_allocator_free(actions->host_allocator, entry);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Now push the ready list to the worker and have it to issue the actions to
  // the GPU.
  entry->ready_list_head = ready_list.head;
  iree_hal_hip_ready_action_slist_push(&actions->working_area.ready_worklist,
                                       entry);

  // We can only overwrite the worker state if the previous state is idle
  // waiting; we cannot overwrite exit related states. so we need to perform
  // atomic compare and exchange here.
  iree_hal_hip_worker_state_t prev_state =
      IREE_HAL_HIP_WORKER_STATE_IDLE_WAITING;
  iree_atomic_compare_exchange_strong_int32(
      &actions->working_area.worker_state, /*expected=*/&prev_state,
      /*desired=*/IREE_HAL_HIP_WORKER_STATE_WORKLOAD_PENDING,
      /*order_succ=*/iree_memory_order_acq_rel,
      /*order_fail=*/iree_memory_order_acquire);
  iree_notification_post(&actions->working_area.state_notification,
                         IREE_ALL_WAITERS);

  // Handle potential error cases from the worker thread.
  if (prev_state == IREE_HAL_HIP_WORKER_STATE_EXIT_ERROR) {
    iree_status_code_t code = iree_atomic_load_int32(
        &actions->working_area.error_code, iree_memory_order_acquire);
    status = iree_status_from_code(code);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Worker routines
//===----------------------------------------------------------------------===//

static bool iree_hal_hip_worker_has_incoming_request_or_error(
    iree_hal_hip_working_area_t* working_area) {
  iree_hal_hip_worker_state_t value = iree_atomic_load_int32(
      &working_area->worker_state, iree_memory_order_acquire);
  return value == IREE_HAL_HIP_WORKER_STATE_WORKLOAD_PENDING ||
         value == IREE_HAL_HIP_WORKER_STATE_EXIT_REQUESTED ||
         value == IREE_HAL_HIP_WORKER_STATE_EXIT_ERROR;
}

static bool iree_hal_hip_worker_committed_exiting(
    iree_hal_hip_working_area_t* working_area) {
  return iree_atomic_load_int32(&working_area->worker_state,
                                iree_memory_order_acquire) ==
         IREE_HAL_HIP_WORKER_STATE_EXIT_COMMITTED;
}

// Processes all ready actions in the given |worklist|.
static iree_status_t iree_hal_hip_worker_process_ready_list(
    iree_allocator_t host_allocator,
    iree_hal_hip_ready_action_slist_t* worklist) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  while (true) {
    iree_hal_hip_atomic_slist_entry_t* entry =
        iree_hal_hip_ready_action_slist_pop(worklist);
    if (!entry) break;

    // Process the current batch of ready actions.
    while (entry->ready_list_head) {
      iree_hal_hip_queue_action_t* action =
          iree_hal_hip_atomic_slist_entry_pop_front(entry);

      switch (action->state) {
        case IREE_HAL_HIP_QUEUE_ACTION_STATE_ALIVE:
          status = iree_hal_hip_pending_queue_actions_issue_execution(action);
          if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
            iree_hal_hip_queue_action_destroy(action);
          }
          break;
        case IREE_HAL_HIP_QUEUE_ACTION_STATE_ZOMBIE:
          iree_hal_hip_pending_queue_actions_issue_cleanup(action);
          break;
      }
      if (IREE_UNLIKELY(!iree_status_is_ok(status))) break;
    }

    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      // Let common destruction path take care of destroying the worklist.
      // When we know all host stream callbacks are done and not touching
      // anything.
      iree_hal_hip_ready_action_slist_push(worklist, entry);
      break;
    }

    iree_allocator_free(host_allocator, entry);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_hal_hip_worker_has_no_pending_host_stream_callbacks(
    iree_hal_hip_working_area_t* working_area) {
  iree_slim_mutex_lock(
      &working_area->host_stream_pending_callbacks_count_mutex);
  bool result = (working_area->host_stream_pending_callbacks_count == 0);
  iree_slim_mutex_unlock(
      &working_area->host_stream_pending_callbacks_count_mutex);
  return result;
}

// Wait for all host stream callbacks to finish.
static void iree_hal_hip_worker_wait_pending_host_stream_callbacks(
    iree_hal_hip_working_area_t* working_area) {
  iree_notification_await(
      &working_area->host_stream_pending_callbacks_count_notification,
      (iree_condition_fn_t)
          iree_hal_hip_worker_has_no_pending_host_stream_callbacks,
      working_area, iree_infinite_timeout());
  // Lock then unlock to make sure that all callbacks are really done.
  // Not even touching the notification.
  iree_slim_mutex_lock(
      &working_area->host_stream_pending_callbacks_count_mutex);
  iree_slim_mutex_unlock(
      &working_area->host_stream_pending_callbacks_count_mutex);
}

// The main function for the ready-list processing worker thread.
static int iree_hal_hip_worker_execute(
    iree_hal_hip_working_area_t* working_area) {
  iree_hal_hip_ready_action_slist_t* worklist = &working_area->ready_worklist;

  while (true) {
    // Block waiting for incoming requests.
    iree_notification_await(
        &working_area->state_notification,
        (iree_condition_fn_t)iree_hal_hip_worker_has_incoming_request_or_error,
        working_area, iree_infinite_timeout());

    // Immediately flip the state to idle waiting if and only if the previous
    // state is workload pending. We do it before processing ready list to make
    // sure that we don't accidentally ignore new workload pushed after done
    // ready list processing but before overwriting the state from this worker
    // thread. Also we don't want to overwrite other exit states. So we need to
    // perform atomic compare and exchange here.
    iree_hal_hip_worker_state_t prev_state =
        IREE_HAL_HIP_WORKER_STATE_WORKLOAD_PENDING;
    iree_atomic_compare_exchange_strong_int32(
        &working_area->worker_state, /*expected=*/&prev_state,
        /*desired=*/IREE_HAL_HIP_WORKER_STATE_IDLE_WAITING,
        /*order_succ=*/iree_memory_order_acq_rel,
        /*order_fail=*/iree_memory_order_acquire);

    int32_t worker_state = iree_atomic_load_int32(&working_area->worker_state,
                                                  iree_memory_order_acquire);
    // Exit if HIP callbacks have posted any errors.
    if (IREE_UNLIKELY(worker_state == IREE_HAL_HIP_WORKER_STATE_EXIT_ERROR)) {
      iree_hal_hip_worker_wait_pending_host_stream_callbacks(working_area);
      return -1;
    }
    // Check if we received request to stop processing and exit this thread.
    bool should_exit =
        (worker_state == IREE_HAL_HIP_WORKER_STATE_EXIT_REQUESTED);

    // Process the ready list. We also want this even requested to exit.
    //
    // TODO: properly handle failure case. Currently we are gonna leak actions.
    // The problem is that the scheduled calls on the callback stream may touch
    // actions. But we can't wait on them to finish, because they may depend on
    // other actions that have not been scheduled because of the failure.
    iree_status_t status = iree_hal_hip_worker_process_ready_list(
        working_area->host_allocator, worklist);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      iree_hal_hip_worker_wait_pending_host_stream_callbacks(working_area);
      iree_hal_hip_post_error_to_worker_state(working_area,
                                              iree_status_code(status));
      return -1;
    }

    if (IREE_UNLIKELY(should_exit && working_area->pending_action_count == 0)) {
      iree_hal_hip_worker_wait_pending_host_stream_callbacks(working_area);
      // Signal that this thread is committed to exit.
      // This state has a priority that is only lower than error exit.
      // A HIP callback may have posted an error, make sure we don't
      // overwrite this error state.
      iree_hal_hip_worker_state_t prev_state =
          IREE_HAL_HIP_WORKER_STATE_EXIT_REQUESTED;
      iree_atomic_compare_exchange_strong_int32(
          &working_area->worker_state, /*expected=*/&prev_state,
          /*desired=*/IREE_HAL_HIP_WORKER_STATE_EXIT_COMMITTED,
          /*order_succ=*/iree_memory_order_acq_rel,
          /*order_fail=*/iree_memory_order_acquire);
      iree_notification_post(&working_area->exit_notification,
                             IREE_ALL_WAITERS);
      return 0;
    }
  }
  return 0;
}
