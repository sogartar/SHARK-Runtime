# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image with latest stuff for building IREE using CMake and testing IREE with
# NVIDIA GPUs.

# To use the host GPUs, `docker run` must be called with the `--gpus all` flag.

# We use .deb files that we host because we have to pin the version and packages
# routinely dissapear from the Ubuntu apt repositories. The versions need to be
# compatible with the host driver (usually <= host driver version).
ARG NVIDIA_GL_DEB="libnvidia-gl-530_530.41.03-0ubuntu0.22.04.2_amd64.deb"
ARG NVIDIA_COMPUTE_DEB="libnvidia-compute-530_530.41.03-0ubuntu0.22.04.2_amd64.deb"
ARG NVIDIA_COMMON_DEB="libnvidia-common-530_530.41.03-0ubuntu0.22.04.2_all.deb"
ARG NVIDIA_EGL_WAYLAND_DEB="libnvidia-egl-wayland1_1.1.9-1.1_amd64.deb"


FROM gcr.io/iree-oss/base-bleeding-edge@sha256:14200dacca3a0f3a66f8aa87c6f64729b83a2eeb403b689c24204074ad157418 AS fetch-nvidia
ARG NVIDIA_COMMON_DEB
ARG NVIDIA_GL_DEB
ARG NVIDIA_COMPUTE_DEB
ARG NVIDIA_EGL_WAYLAND_DEB

WORKDIR /fetch-nvidia
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMMON_DEB}"
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_GL_DEB}"
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMPUTE_DEB}"
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_EGL_WAYLAND_DEB}"


# Set up the image and working directory by inheriting the base-bleeding-edge
# CMake configuration.
# Note that we don't start from NVIDIA's docker base:
# - nvidia/cuda (https://hub.docker.com/r/nvidia/cuda):
#     it's.. for CUDA.
# - nvidia/vulkan (https://hub.docker.com/r/nvidia/vulkan):
#      does not support Ubuntu 22.04.
# This allows to share configuration with base CMake.
FROM gcr.io/iree-oss/base-bleeding-edge@sha256:14200dacca3a0f3a66f8aa87c6f64729b83a2eeb403b689c24204074ad157418 AS final
ARG NVIDIA_COMMON_DEB
ARG NVIDIA_GL_DEB
ARG NVIDIA_COMPUTE_DEB
ARG NVIDIA_EGL_WAYLAND_DEB

COPY --from=fetch-nvidia \
  "/fetch-nvidia/${NVIDIA_COMMON_DEB}" \
  "/fetch-nvidia/${NVIDIA_GL_DEB}" \
  "/fetch-nvidia/${NVIDIA_COMPUTE_DEB}" \
  "/fetch-nvidia/${NVIDIA_EGL_WAYLAND_DEB}" \
  /tmp/

# The local .deb files have dependencies that requires apt-get update to locate.
RUN apt-get update \
  && apt-get -y install "/tmp/${NVIDIA_COMMON_DEB}" \
  "/tmp/${NVIDIA_GL_DEB}" \
  "/tmp/${NVIDIA_COMPUTE_DEB}" \
  "/tmp/${NVIDIA_EGL_WAYLAND_DEB}"

# Install the CUDA SDK
RUN wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb \
  && dpkg --install cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb \
  && cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
  && apt-get update \
  && apt-get -y install cuda-toolkit-12-1

# Adding CUDA binaries to Path
ENV PATH=${PATH}:/usr/local/cuda/bin/

# Collectives: NCCL+OpenMPL
RUN sudo apt-get -y install libnccl-dev=2.18.1-1+cuda12.1 libopenmpi-dev
