/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Translation unit for the single TE<->PyTorch binary boundary.
//
// The facade helpers (GetATenDType, GetTransformerEngineDType, new_cuda_tensor)
// are defined `inline` in torch_backend.h: they sit on the per-tensor
// Python<->C++ marshalling path, so keeping them inlinable matches the
// pre-facade codegen. This .cpp is intentionally empty; it exists as the home
// for any future non-inline boundary code (and, together with torch_backend.h,
// is the only place allowed to name at::/c10::/torch:: symbols).

#include "torch_backend.h"
