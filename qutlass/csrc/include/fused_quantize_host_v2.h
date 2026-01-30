/*
 * Copyright (C) 2025 Moiz A. Yousufi (moiz.yousufi@gatech.edu). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <common.h>

/**
 * CUTLASS 4.x Fused Quantization API
 *
 * These functions support ARBITRARY K dimensions (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, etc.)
 * No more K≤256 limitation!
 *
 * Architecture: Blackwell SM120 (RTX 5090) and SM100 (B200)
 */
namespace QUTLASS_V2 {

/**
 * Fused Hadamard rotation + Quest quantization (variance-based scaling)
 *
 * Computes: D_fp4, D_scale = Quantize(A @ H^T)
 *   where scale = sqrt(var) * (2.92247856 / 6) + epsilon
 *
 * @param D       Output tensor: packed E2M1 values [M, K/2]
 * @param D_sf    Output tensor: scale factors [M, K/32] (one per 32-element group)
 * @param A       Input tensor: BF16 values [M, K]
 * @param H       Hadamard rotation matrix: BF16 [K, K]
 *
 * Supports K = 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, ...
 */
void fusedQuantizeMxQuest_v2(torch::Tensor&       D,
                             torch::Tensor&       D_sf,
                             torch::Tensor const& A,
                             torch::Tensor const& H);

/**
 * Fused Hadamard rotation + AbsMax quantization (maximum absolute value scaling)
 *
 * Computes: D_fp4, D_scale = Quantize(A @ H^T)
 *   where scale = max(|x|) + epsilon
 *
 * @param D       Output tensor: packed E2M1 values [M, K/2]
 * @param D_sf    Output tensor: scale factors [M, K/32] (one per 32-element group)
 * @param A       Input tensor: BF16 values [M, K]
 * @param H       Hadamard rotation matrix: BF16 [K, K]
 *
 * Supports K = 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, ...
 */
void fusedQuantizeMxAbsMax_v2(torch::Tensor&       D,
                              torch::Tensor&       D_sf,
                              torch::Tensor const& A,
                              torch::Tensor const& H);

}  // namespace QUTLASS_V2
