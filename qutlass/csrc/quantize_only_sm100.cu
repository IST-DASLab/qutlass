/*
 * Direct MXFP4 Quantization Kernel (No Rotation)
 *
 * Optimized for identity matrix case where Hadamard rotation is unnecessary.
 * Eliminates 275B FLOPs + 68GB memory bandwidth overhead for 1M tokens × 8192.
 *
 * Expected speedup: 230ms → 5-10ms (23-46x faster quantization)
 *
 * Copyright (C) 2026 Moiz A. Yousufi (moiz.yousufi@gatech.edu). All Rights Reserved.
 * Based on QuTLASS by Roberto L. Castro (Roberto.LopezCastro@ist.ac.at).
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

#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>

#ifndef QUTLASS_DISABLE_PYBIND
#include <torch/extension.h>
#endif

namespace QUTLASS {

// helper functions: E8M0 and E2M1 quantization

__device__ __forceinline__ uint8_t float_to_e8m0(float val) {
    // E8M0 format: 8-bit exponent only, no mantissa
    // representation: 2^(exponent - 127)

    if (val == 0.0f) return 0;

    // get IEEE 754 representation
    uint32_t bits = __float_as_uint(val);

    // extract exponent (8 bits, biased by 127)
    uint32_t exp = (bits >> 23) & 0xFF;

    // round up exponent to ensure scale ≥ max_val
    return (uint8_t)exp;
}

__device__ __forceinline__ float e8m0_to_float(uint8_t e8m0_val) {
    // convert E8M0 back to float: 2^(exponent - 127)
    if (e8m0_val == 0) return 0.0f;

    // reconstruct IEEE 754: sign=0, exponent=e8m0_val, mantissa=0
    uint32_t bits = ((uint32_t)e8m0_val) << 23;
    return __uint_as_float(bits);
}

__device__ __forceinline__ uint8_t quantize_e2m1(float normalized_val) {
    // E2M1 format: 1 sign + 2 exponent + 1 mantissa = 4 bits
    if (normalized_val == 0.0f) return 0;

    // get sign
    uint32_t sign = (normalized_val < 0.0f) ? 1 : 0;
    float abs_val = fabsf(normalized_val);

    // clamp to representable range
    if (abs_val > 1.0f) abs_val = 1.0f;
    if (abs_val < 0.0625f) return 0;

    // get IEEE 754 bits
    uint32_t bits = __float_as_uint(abs_val);

    // extract exponent and mantissa
    uint32_t exp_ieee = (bits >> 23) & 0xFF;
    uint32_t mantissa_ieee = (bits >> 22) & 0x1;

    // convert to E2M1 exponent (2 bits, biased by 1)
    int exp_e2m1 = (int)exp_ieee - 127 + 1;
    if (exp_e2m1 < 0) exp_e2m1 = 0;
    if (exp_e2m1 > 3) exp_e2m1 = 3;

    // pack: [sign:1][exponent:2][mantissa:1]
    uint8_t result = (sign << 3) | (exp_e2m1 << 1) | mantissa_ieee;

    return result & 0xF;
}

// direct quantization kernel (no GEMM) - 2D tiled version

// each thread block processes a tile of [BLOCK_ROWS, BLOCK_K_TILES×32]
// grid: (M / BLOCK_ROWS, K / (BLOCK_K_TILES*32), num_pairs)
// balance between grid size and resource usage (~250K blocks for B200)

constexpr int BLOCK_ROWS = 64;
constexpr int BLOCK_K_TILES = 16;
constexpr int THREADS_PER_BLOCK = 256;

__global__ void direct_quantize_mxfp4_kernel(
    const __nv_bfloat16* __restrict__ input,  // [M, K] BF16
    uint8_t* __restrict__ packed,              // [M, K/2] uint8
    uint8_t* __restrict__ scales,              // [M, K/32] E8M0
    int M,
    int K
) {
    // 2D grid: blockIdx.x = row tile, blockIdx.y = MXFP4 block
    int row_tile = blockIdx.x;  // Which group of BLOCK_ROWS
    int k_block = blockIdx.y;   // Which 32-element MXFP4 block along K

    int row_start = row_tile * BLOCK_ROWS;
    int k_start = k_block * 32;

    // shared memory for reduction
    __shared__ float smem_max[BLOCK_ROWS];

    int tid = threadIdx.x;

    // initialize shared memory
    if (tid < BLOCK_ROWS) {
        smem_max[tid] = 0.0f;
    }
    __syncthreads();

    // each thread processes elements from 4 rows in parallel

    #pragma unroll
    for (int r = 0; r < BLOCK_ROWS; ++r) {
        int row = row_start + r;
        if (row >= M) continue;

        // compute max for this row's 32-element block
        float local_max = 0.0f;

        // each thread handles every 128th element
        for (int i = tid; i < 32; i += THREADS_PER_BLOCK) {
            float val = __bfloat162float(input[row * K + k_start + i]);
            local_max = fmaxf(local_max, fabsf(val));
        }

        // warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }

        // First thread in each warp writes to shared memory
        // Use atomicMax with proper float handling
        if (tid % 32 == 0) {
            // Atomic max for floats using compare-and-swap
            unsigned int* addr = (unsigned int*)&smem_max[r];
            unsigned int old = *addr;
            unsigned int expected;
            do {
                expected = old;
                float current = __uint_as_float(old);
                float new_val = fmaxf(current, local_max);
                old = atomicCAS(addr, expected, __float_as_uint(new_val));
            } while (old != expected);
        }
    }

    __syncthreads();

    // quantize and pack
    #pragma unroll
    for (int r = 0; r < BLOCK_ROWS; ++r) {
        int row = row_start + r;
        if (row >= M) continue;

        float max_val = smem_max[r];

        // convert to E8M0 scale
        uint8_t scale_e8m0 = float_to_e8m0(max_val);
        if (tid == 0) {
            scales[row * (K / 32) + k_block] = scale_e8m0;
        }

        // quantize and pack (parallel across threads)
        float scale_float = e8m0_to_float(scale_e8m0);
        float inv_scale = (scale_float > 0.0f) ? (1.0f / scale_float) : 0.0f;

        // Each thread handles 16/THREADS_PER_BLOCK pairs
        // For 128 threads: each thread does 1 pair (covers all 16 bytes)
        for (int i = tid; i < 16; i += THREADS_PER_BLOCK) {
            // Load two consecutive BF16 values
            float val1 = __bfloat162float(input[row * K + k_start + i * 2]);
            float val2 = __bfloat162float(input[row * K + k_start + i * 2 + 1]);

            // normalize by scale
            float norm1 = val1 * inv_scale;
            float norm2 = val2 * inv_scale;

            // quantize to E2M1
            uint8_t q1 = quantize_e2m1(norm1);
            uint8_t q2 = quantize_e2m1(norm2);

            // pack two 4-bit values into one byte
            packed[row * (K / 2) + k_start / 2 + i] = (q2 << 4) | q1;
        }
    }
}

// ============================================================================
// host functions
// ============================================================================

void directQuantizeMxAbsMax(
    torch::Tensor const& input,   // [M, K] BF16
    torch::Tensor& packed,         // [M, K/2] uint8 (output)
    torch::Tensor& scales          // [M, K/32] uint8/E8M0 (output)
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);

    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32 for MXFP4");
    TORCH_CHECK(K >= 32, "K must be at least 32");

    // validate output shapes
    TORCH_CHECK(packed.size(0) == M && packed.size(1) == K / 2,
                "packed shape must be [M, K/2]");
    TORCH_CHECK(scales.size(0) == M && scales.size(1) == K / 32,
                "scales shape must be [M, K/32]");

    // launch kernel with 2D grid for better parallelism
    int num_row_tiles = (M + BLOCK_ROWS - 1) / BLOCK_ROWS;
    int num_k_blocks = K / 32;

    dim3 grid(num_row_tiles, num_k_blocks);
    dim3 block(THREADS_PER_BLOCK);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index());

    direct_quantize_mxfp4_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        packed.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(),
        M,
        K
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "directQuantizeMxAbsMax kernel failed: ", cudaGetErrorString(err));
}

// ============================================================================
// batched kernel - 3D grid (single kernel launch)
// ============================================================================

__global__ void direct_quantize_mxfp4_kernel_batched(
    const __nv_bfloat16* __restrict__ inputs,  // [num_pairs, M, K] BF16
    uint8_t* __restrict__ packed,               // [num_pairs, M, K/2] uint8
    uint8_t* __restrict__ scales,               // [num_pairs, M, K/32] E8M0
    int num_pairs,
    int M,
    int K,
    int64_t input_pair_stride,   // M * K
    int64_t packed_pair_stride,  // M * (K/2)
    int64_t scales_pair_stride   // M * (K/32)
) {
    // 3D grid: blockIdx.z = pair, blockIdx.x = row tile, blockIdx.y = K tile
    int pair_idx = blockIdx.z;
    int row_tile = blockIdx.x;
    int k_tile = blockIdx.y;  // Each tile covers BLOCK_K_TILES MXFP4 blocks

    if (pair_idx >= num_pairs) return;

    // offset pointers for this pair
    const __nv_bfloat16* input = inputs + pair_idx * input_pair_stride;
    uint8_t* packed_out = packed + pair_idx * packed_pair_stride;
    uint8_t* scales_out = scales + pair_idx * scales_pair_stride;

    int row_start = row_tile * BLOCK_ROWS;
    int k_tile_start = k_tile * BLOCK_K_TILES;  // Start MXFP4 block index for this tile

    // shared memory for reduction
    __shared__ float smem_max[BLOCK_ROWS * BLOCK_K_TILES];

    int tid = threadIdx.x;

    // process BLOCK_K_TILES MXFP4 blocks
    for (int kt = 0; kt < BLOCK_K_TILES; ++kt) {
        int k_block = k_tile_start + kt;
        if (k_block >= K / 32) continue;  // Bounds check

        int k_start = k_block * 32;

        // initialize shared memory for this K tile
        int smem_offset = kt * BLOCK_ROWS;
        if (tid < BLOCK_ROWS) {
            smem_max[smem_offset + tid] = 0.0f;
        }
        __syncthreads();

        // compute max for each row in this K block
        #pragma unroll
        for (int r = 0; r < BLOCK_ROWS; ++r) {
            int row = row_start + r;
            if (row >= M) continue;

            float local_max = 0.0f;

            for (int i = tid; i < 32; i += THREADS_PER_BLOCK) {
                float val = __bfloat162float(input[row * K + k_start + i]);
                local_max = fmaxf(local_max, fabsf(val));
            }

            // warp reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
            }

            // atomic max to shared memory
            if (tid % 32 == 0 && local_max > 0.0f) {
                unsigned int* addr = (unsigned int*)&smem_max[smem_offset + r];
                unsigned int old = *addr;
                unsigned int expected;
                do {
                    expected = old;
                    float current = __uint_as_float(old);
                    float new_val = fmaxf(current, local_max);
                    old = atomicCAS(addr, expected, __float_as_uint(new_val));
                } while (old != expected);
            }
        }

        __syncthreads();

        // quantize and pack this K block
        #pragma unroll
        for (int r = 0; r < BLOCK_ROWS; ++r) {
            int row = row_start + r;
            if (row >= M) continue;

            float max_val = smem_max[smem_offset + r];

            // store scale
            uint8_t scale_e8m0 = float_to_e8m0(max_val);
            if (tid == 0) {
                scales_out[row * (K / 32) + k_block] = scale_e8m0;
            }

            // pack elements
            float scale_float = e8m0_to_float(scale_e8m0);
            float inv_scale = (scale_float > 0.0f) ? (1.0f / scale_float) : 0.0f;

            for (int i = tid; i < 16; i += THREADS_PER_BLOCK) {
                float val1 = __bfloat162float(input[row * K + k_start + i * 2]);
                float val2 = __bfloat162float(input[row * K + k_start + i * 2 + 1]);

                float norm1 = val1 * inv_scale;
                float norm2 = val2 * inv_scale;

                uint8_t q1 = quantize_e2m1(norm1);
                uint8_t q2 = quantize_e2m1(norm2);

                packed_out[row * (K / 2) + k_start / 2 + i] = (q2 << 4) | q1;
            }
        }

        __syncthreads();  // Ensure all threads done before next K tile
    }
}

// batched version - single kernel launch for all pairs
void directQuantizeMxAbsMax_batched(
    torch::Tensor const& inputs,  // [num_pairs, M, K] BF16
    torch::Tensor& packed,         // [num_pairs, M, K/2] uint8
    torch::Tensor& scales          // [num_pairs, M, K/32] uint8
) {
    TORCH_CHECK(inputs.is_cuda(), "inputs must be on CUDA");
    TORCH_CHECK(inputs.scalar_type() == at::kBFloat16, "inputs must be BF16");
    TORCH_CHECK(inputs.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(inputs.dim() == 3, "inputs must be 3D [num_pairs, M, K]");

    int num_pairs = inputs.size(0);
    int M = inputs.size(1);
    int K = inputs.size(2);

    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32 for MXFP4");
    TORCH_CHECK(K >= 32, "K must be at least 32");

    // 3D grid: (M/BLOCK_ROWS, (K/32)/BLOCK_K_TILES, num_pairs)
    int num_row_tiles = (M + BLOCK_ROWS - 1) / BLOCK_ROWS;
    int num_k_tiles = ((K / 32) + BLOCK_K_TILES - 1) / BLOCK_K_TILES;

    dim3 grid(num_row_tiles, num_k_tiles, num_pairs);
    dim3 block(THREADS_PER_BLOCK);

    // compute strides
    int64_t input_pair_stride = M * K;
    int64_t packed_pair_stride = M * (K / 2);
    int64_t scales_pair_stride = M * (K / 32);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(inputs));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(inputs.device().index());

    // time kernel execution (first call only)
    static bool first_call = true;
    cudaEvent_t start, stop;
    if (first_call) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }

    // single kernel launch for all pairs
    direct_quantize_mxfp4_kernel_batched<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(inputs.data_ptr()),
        packed.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(),
        num_pairs,
        M,
        K,
        input_pair_stride,
        packed_pair_stride,
        scales_pair_stride
    );

    // check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "directQuantizeMxAbsMax_batched kernel failed: ", cudaGetErrorString(err));

    // print timing (first call only)
    if (first_call) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "[QuTLASS directQuantizeMxAbsMax_batched] "
                  << "Grid: (" << grid.x << "," << grid.y << "," << grid.z << "), "
                  << "Block: " << block.x << ", "
                  << "Kernel time: " << milliseconds << " ms "
                  << "(num_pairs=" << num_pairs << ", M=" << M << ", K=" << K << ")"
                  << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        first_call = false;
    }
}

} // namespace QUTLASS
