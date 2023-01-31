# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

################################################################################
# Package
################################################################################

iree_py_library(
  NAME
    testing
  SRCS
    "iree/testing.py"
)

################################################################################
# Install
################################################################################

set(_INSTALL_DIR "python_packages/iree_runtime")
set(_INSTALL_COMPONENT "IreePythonPackage-testing")

set(_INSTALL_DIR "python_packages/iree_testing")
set(_INSTALL_COMPONENT "IreePythonPackage-testing")

install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/iree/testing.py"
  COMPONENT "${_INSTALL_COMPONENT}"
  DESTINATION "${_INSTALL_DIR}/iree"
)
