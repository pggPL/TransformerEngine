/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "util.h"
#include "ATen/cuda/CUDAContextLight.h"

bool supports_fp8_transposes() {
  int major = at::cuda::getCurrentDeviceProperties()->major;
  return major >= 10;
}
