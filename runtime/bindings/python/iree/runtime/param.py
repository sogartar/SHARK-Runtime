# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""ParameterIndex and numpy interop."""

import json
import numpy as np
import warnings

from ._binding import (
    ParameterIndex,
    ParameterIndexEntry,
)

__all__ = [
    "parameter_index_add_numpy_ndarray",
    "parameter_index_entry_as_numpy_flat_ndarray",
    "parameter_index_entry_as_numpy_ndarray",
]

_DTYPE_TO_NAME = (
    (np.float16, "float16"),
    (np.float32, "float32"),
    (np.float64, "float64"),
    (np.int32, "int32"),
    (np.int64, "int64"),
    (np.int16, "int16"),
    (np.int8, "int8"),
    (np.uint32, "uint32"),
    (np.uint64, "uint64"),
    (np.uint16, "uint16"),
    (np.uint8, "uint8"),
    (np.bool_, "bool"),
    (np.complex64, "complex64"),
    (np.complex128, "complex128"),
)

_NAME_TO_DTYPE: dict[str, np.dtype] = {
    name: np_dtype for np_dtype, name in _DTYPE_TO_NAME
}

_metadata_prefix = "PYTORCH:"


def _make_tensor_metadata(t: np.ndarray) -> str:
    """Makes a tensor metadata blob that can be used to reconstruct the tensor."""
    dtype = t.dtype
    dtype_name = _DTYPE_TO_NAME[dtype]
    dtype_desc = {
        "class_name": type(dtype).__name__,
        "is_complex": dtype.is_complex,
        "is_floating_point": dtype.is_floating_point,
        "is_signed": dtype.is_signed,
        "itemsize": dtype.itemsize,
    }
    d = {
        "type": "Tensor",
        "dtype": dtype_name,
        "shape": list(t.shape),
        "dtype_desc": dtype_desc,
    }
    encoded = f"{_metadata_prefix}{json.dumps(d)}"
    return encoded


def parameter_index_add_numpy_ndarray(
    index: ParameterIndex, name: str, array: np.ndarray
):
    """Adds an named array to the index."""
    metadata = _make_tensor_metadata(tensor)
    # 0d arrays are special in both torch/numpy in different ways that makes
    # it hard to reliably get a memory view of their contents. Since we
    # know that 0d is always small, we just force a copy when in numpy
    # land and that seems to get it on the happy path.
    # See: https://github.com/iree-org/iree-turbine/issues/29
    if len(tensor.shape) == 0:
        flat_array_np = tensor.detach().cpu().numpy().copy()
        host_array = flat_array_np
    else:
        flat_array = tensor.detach().flatten().contiguous().cpu().view(torch.uint8)
        host_array = flat_array.numpy()
    self._index.add_buffer(name, host_array, metadata=metadata)


def parameter_index_entry_as_numpy_flat_ndarray(
    index_entry: ParameterIndexEntry,
) -> np.ndarray:
    """Accesses the contents as a uint8 flat tensor.

    If it is a splat, then the tensor will be a view of the splat pattern.

    Raises a ValueError on unsupported entries.
    """
    if index_entry.is_file:
        wrapper = np.array(index_entry.file_view, copy=False)
    elif index_entry.is_splat:
        wrapper = np.array(index_entry.splat_pattern, copy=True)
    else:
        raise ValueError(f"Unsupported ParameterIndexEntry: {index_entry}")

    return wrapper


def parameter_index_entry_as_numpy_ndarray(
    index_entry: ParameterIndexEntry,
) -> np.ndarray:
    """Returns a tensor viewed with appropriate shape/dtype from metadata.

    Raises a ValueError if unsupported.
    """

    # Decode metadata.
    metadata = index_entry.metadata.decode()
    if not metadata.startswith(metadata_prefix):
        raise ValueError(
            f"No metadata for parameter entry {index_entry.key}: Cannot convert to tensor"
        )
    metadata = metadata[len(metadata_prefix) :]
    d = json.loads(metadata)
    try:
        type_name = d["type"]
        if d["type"] != "Tensor":
            raise ValueError(
                f"Metadata for parameter entry {index_entry.key} is not a Tensor ('{type_name}')"
            )
        dtype_name = d["dtype"]
        shape = d["shape"]
    except KeyError as e:
        raise ValueError(f"Bad metadata for parameter entry {index_entry.key}") from e

    # Unpack/validate.
    try:
        dtype = _NAME_TO_DTYPE[dtype_name]
    except KeyError:
        raise ValueError(f"Unknown dtype name '{dtype_name}'")
    try:
        shape = [int(d) for d in shape]
    except ValueError as e:
        raise ValueError(f"Illegal shape for parameter entry {index_entry.key}") from e

    t = parameter_index_entry_as_numpy_flat_ndarray(index_entry)
    return t.view(dtype=dtype).reshape(shape)
