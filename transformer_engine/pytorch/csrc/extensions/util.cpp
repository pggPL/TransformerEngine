/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "util.h"
#include "ATen/cuda/CUDAContextLight.h"

bool non_tn_fp8_gemm_supported() {
  int major = at::cuda::getCurrentDeviceProperties()->major;
  return major >= 10;
}
