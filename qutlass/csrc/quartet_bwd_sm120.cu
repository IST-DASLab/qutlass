/*
 * Copyright (C) 2025 Roberto L. Castro (Roberto.LopezCastro@ist.ac.at). All Rights Reserved.
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

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <mma.h>

#ifndef QUTLASS_DISABLE_PYBIND
#include <torch/extension.h>
#endif

#include <backward_host.h>

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<__half> {
    static inline __device__ float toFloat(__half val) {
        return __half2float(val);
    }

    static inline __device__ __half fromFloat(float val) {
        return __float2half_rn(val);
    }

    static inline __device__ void e2m1_to_vec(uint32_t* array, uint32_t x)
    {
        asm volatile(
            "{\n"
            ".reg .b8 byte0;\n"
            ".reg .b8 byte1;\n"
            ".reg .b8 byte2;\n"
            ".reg .b8 byte3;\n"
            "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
            "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
            "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
            "cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
            "cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
            "}"
            : "=r"(array[0]), "=r"(array[1]), "=r"(array[2]), "=r"(array[3])
            : "r"(x));
    }
};

template <>
struct TypeTraits<__nv_bfloat16> {
    static inline __device__ float toFloat(__nv_bfloat16 val) {
        return __bfloat162float(val);
    }

    static inline __device__ __nv_bfloat16 fromFloat(float val) {
        return __float2bfloat16_rn(val);
    }

    static inline __device__ void e2m1_to_vec(uint32_t* array, uint32_t x)
    {
        asm volatile(
            "{\n"
            ".reg .b8 byte0;\n"
            ".reg .b8 byte1;\n"
            ".reg .b8 byte2;\n"
            ".reg .b8 byte3;\n"
            ".reg .b32 dword0;\n"
            ".reg .b32 dword1;\n"
            ".reg .b32 dword2;\n"
            ".reg .b32 dword3;\n"
            ".reg .b16 word0;\n"
            ".reg .b16 word1;\n"
            ".reg .b16 word2;\n"
            ".reg .b16 word3;\n"
            ".reg .b16 word4;\n"
            ".reg .b16 word5;\n"
            ".reg .b16 word6;\n"
            ".reg .b16 word7;\n"
            "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
            "cvt.rn.f16x2.e2m1x2 dword0, byte0;\n"
            "cvt.rn.f16x2.e2m1x2 dword1, byte1;\n"
            "cvt.rn.f16x2.e2m1x2 dword2, byte2;\n"
            "cvt.rn.f16x2.e2m1x2 dword3, byte3;\n"
            "mov.b32 {word0, word1}, dword0;\n"
            "mov.b32 {word2, word3}, dword1;\n"
            "mov.b32 {word4, word5}, dword2;\n"
            "mov.b32 {word6, word7}, dword3;\n"
            "cvt.rn.bf16.f16 word0, word0;\n"
            "cvt.rn.bf16.f16 word1, word1;\n"
            "cvt.rn.bf16.f16 word2, word2;\n"
            "cvt.rn.bf16.f16 word3, word3;\n"
            "cvt.rn.bf16.f16 word4, word4;\n"
            "cvt.rn.bf16.f16 word5, word5;\n"
            "cvt.rn.bf16.f16 word6, word6;\n"
            "cvt.rn.bf16.f16 word7, word7;\n"
            "mov.b32 %0, {word0, word1};\n"
            "mov.b32 %1, {word2, word3};\n"
            "mov.b32 %2, {word4, word5};\n"
            "mov.b32 %3, {word6, word7};\n"
            "}"
            : "=r"(array[0]), "=r"(array[1]), "=r"(array[2]), "=r"(array[3])
            : "r"(x));
    }

    static inline __device__ void e8m0_to_vec(uint32_t* array, uint32_t x)
    {
        asm volatile(
            "{\n"
            ".reg .b16 word0;\n"
            ".reg .b16 word1;\n"
            "mov.b32 {word0, word1}, %2;\n"
            "cvt.rn.bf16x2.ue8m0x2 %0, word0;\n"
            "cvt.rn.bf16x2.ue8m0x2 %1, word1;\n"
            "}"
            : "=r"(array[0]), "=r"(array[1])
            : "r"(x));
    }
};


inline __device__ uint32_t fp32_vec_to_e2m1(float* array)
{
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
    return val;
}


template<typename InputDtype>
__global__ void cvt_e2m1_cuda_kernel(
    uint32_t* __restrict__ y_ptr,
    const InputDtype* __restrict__ x_ptr,
    const int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size / 8) {
        return;
    }

    float x_fp32[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      x_fp32[i] = TypeTraits<InputDtype>::toFloat(x_ptr[idx * 8 + i]);
    }

    y_ptr[idx] = fp32_vec_to_e2m1((float*) x_fp32);
}


int cvt_bf16_e2m1_cuda(
    void* y_ptr,
    const void* x_ptr,
    const int size,
    cudaStream_t stream
)
{
    using InputDtype = __nv_bfloat16;
    const int group_size = 8, blockSize = 256, gridSize  = (size + blockSize * group_size - 1) / (blockSize * group_size);
    cvt_e2m1_cuda_kernel<InputDtype><<<gridSize, blockSize, 0, stream>>>(
        (uint32_t*) y_ptr,
        (InputDtype*) x_ptr,
        size
    );
    return 0;
}


template<typename InputDtype>
__global__ void cvt_e2m1_cuda_kernel(
    uint32_t* __restrict__ y_ptr,
    const uint32_t* __restrict__ x_ptr,
    const int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size / 8) {
        return;
    }

    uint32_t y_data[4];

    TypeTraits<InputDtype>::e2m1_to_vec((uint32_t*) y_data, x_ptr[idx]);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        y_ptr[idx * 4 + i] = y_data[i];
    }
}


int cvt_e2m1_bf16_cuda(
    void* y_ptr,
    const void* x_ptr,
    const int size,
    cudaStream_t stream
)
{
    using InputDtype = __nv_bfloat16;
    const int group_size = 8, blockSize = 256, gridSize  = (size + blockSize * group_size - 1) / (blockSize * group_size);
    cvt_e2m1_cuda_kernel<InputDtype><<<gridSize, blockSize, 0, stream>>>(
        (uint32_t*) y_ptr,
        (uint32_t*) x_ptr,
        size
    );
    return 0;
}


template<typename InputDtype>
__global__ void quantize_g32t_cuda_kernel(const InputDtype *A,
                                          const InputDtype *B,
                                          uint32_t *xh_e2m1_ptr,
                                          uint8_t *xh_e8m0_ptr,
                                          const int size_m,
                                          const int size_n,
                                          const int size_b)
{
    const int tile_size_m = 8, tile_size_n = 32, tile_size_k = 16, size_k = 32;

    __shared__ InputDtype shB[size_k * tile_size_n];

    const InputDtype *shB_src = B + size_k * threadIdx.x;
    InputDtype *shB_dst = shB + size_k * threadIdx.x;
    #pragma unroll
    for (int col = 0; col < size_k; ++col) {
        shB_dst[col] = shB_src[col];
    }

    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tile_size_m, tile_size_n, tile_size_k, InputDtype, nvcuda::wmma::col_major> frag_a1, frag_a2;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tile_size_m, tile_size_n, tile_size_k, InputDtype, nvcuda::wmma::row_major> frag_b1, frag_b2;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tile_size_m, tile_size_n, tile_size_k, float> frag_c;

    nvcuda::wmma::fill_fragment(frag_c, 0.f);

    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int num_tiles_m = size_m / tile_size_m,
              num_tiles_n = size_n / tile_size_n,
              num_tiles_mn = num_tiles_m * num_tiles_n;

    const int block_size_m = 128, block_size_n = 128;

    const int num_blocks_m = (size_m + block_size_m - 1) / block_size_m,
              num_blocks_n = (size_n + block_size_n - 1) / block_size_n,
              num_tiles_in_block_m = block_size_m / tile_size_m,
              num_tiles_in_block_n = block_size_n / tile_size_n,
              num_tiles_in_block_mn = num_tiles_in_block_m * num_tiles_in_block_n;

    const int tile_c_idx_b = idx / num_tiles_mn,
              tile_c_idx_block = idx % num_tiles_mn / num_tiles_in_block_mn,
              tile_c_idx_block_m = tile_c_idx_block / num_blocks_n,
              tile_c_idx_block_n = tile_c_idx_block % num_blocks_n,
              tile_c_idx_in_block = idx % num_tiles_in_block_mn,
              tile_c_idx_in_block_m = tile_c_idx_in_block % num_tiles_in_block_m,
              tile_c_idx_in_block_n = tile_c_idx_in_block / num_tiles_in_block_m,
              tile_c_idx_m = tile_c_idx_block_m * num_tiles_in_block_m + tile_c_idx_in_block_m,
              tile_c_idx_n = tile_c_idx_block_n * num_tiles_in_block_n + tile_c_idx_in_block_n;

    const InputDtype *tile_a1 = A + tile_c_idx_b * size_n * size_m + tile_c_idx_n * size_k * size_m + tile_c_idx_m * tile_size_m,
                     *tile_a2 = tile_a1 + tile_size_k * size_m,
                     *tile_b1 = shB,
                     *tile_b2 = tile_b1 + tile_size_k * size_k;

    nvcuda::wmma::load_matrix_sync(frag_a1, tile_a1, size_m);
    nvcuda::wmma::load_matrix_sync(frag_b1, tile_b1, size_k);

    nvcuda::wmma::mma_sync(frag_c, frag_a1, frag_b1, frag_c);

    nvcuda::wmma::load_matrix_sync(frag_a2, tile_a2, size_m);
    nvcuda::wmma::load_matrix_sync(frag_b2, tile_b2, size_k);

    nvcuda::wmma::mma_sync(frag_c, frag_a2, frag_b2, frag_c);

    __shared__ float mat_c[tile_size_m][tile_size_n];
    nvcuda::wmma::store_matrix_sync(&mat_c[0][0], frag_c, tile_size_n, nvcuda::wmma::mem_row_major);

    if (threadIdx.x < tile_size_m) {
        float scale = 0.f;
        #pragma unroll
        for(int i = 0; i < tile_size_n; ++i) {
            float c_val = mat_c[threadIdx.x][i];
            scale = max(abs(c_val), scale);
        }
        reinterpret_cast<uint32_t&>(scale) = (reinterpret_cast<uint32_t&>(scale) /*+ 0x7f000000*/) & 0x7f800000;
        xh_e8m0_ptr[((tile_c_idx_b * num_tiles_m + tile_c_idx_m) * tile_size_m + threadIdx.x) * num_tiles_n + tile_c_idx_n] = reinterpret_cast<uint32_t&>(scale) >> 23;
        #pragma unroll
        for(int i = 0; i < tile_size_n; ++i) {
            mat_c[threadIdx.x][i] *= 3.f / scale;
        }
    }

    __syncthreads();

    const int num_threads_per_row = blockDim.x / tile_size_m;
    xh_e2m1_ptr[(((tile_c_idx_b * num_tiles_m + tile_c_idx_m) * tile_size_m + threadIdx.x / num_threads_per_row) * num_tiles_n + tile_c_idx_n) * num_threads_per_row + threadIdx.x % num_threads_per_row] = fp32_vec_to_e2m1((float*) mat_c + tile_size_m * threadIdx.x);
}

template<typename InputDtype>
__global__ void quantize_g32qt_cuda_kernel(const uint32_t *x_e2m1_ptr,
                                           const uint8_t *x_e8m0_ptr,
                                           const InputDtype *B,
                                           const float *alpha,
                                           uint32_t *xh_e2m1_ptr,
                                           uint8_t *xh_e8m0_ptr,
                                           const int size_m,
                                           const int size_n,
                                           const int size_b)
{
    const int tile_size_m = 8, tile_size_n = 32, tile_size_k = 16, size_k = 32;

    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int num_tiles_m = size_m / tile_size_m,
              num_tiles_n = size_n / tile_size_n,
              num_tiles_mn = num_tiles_m * num_tiles_n;

    const int block_size_m = 128, block_size_n = 128;

    const int num_blocks_m = (size_m + block_size_m - 1) / block_size_m,
              num_blocks_n = (size_n + block_size_n - 1) / block_size_n,
              num_tiles_in_block_m = block_size_m / tile_size_m,
              num_tiles_in_block_n = block_size_n / tile_size_n,
              num_tiles_in_block_mn = num_tiles_in_block_m * num_tiles_in_block_n;

    const int tile_c_idx_b = idx / num_tiles_mn,
              tile_c_idx_block = idx % num_tiles_mn / num_tiles_in_block_mn,
              tile_c_idx_block_m = tile_c_idx_block / num_blocks_n,
              tile_c_idx_block_n = tile_c_idx_block % num_blocks_n,
              tile_c_idx_in_block = idx % num_tiles_in_block_mn,
              tile_c_idx_in_block_m = tile_c_idx_in_block % num_tiles_in_block_m,
              tile_c_idx_in_block_n = tile_c_idx_in_block / num_tiles_in_block_m,
              tile_c_idx_m = tile_c_idx_block_m * num_tiles_in_block_m + tile_c_idx_in_block_m,
              tile_c_idx_n = tile_c_idx_block_n * num_tiles_in_block_n + tile_c_idx_in_block_n;

    __shared__ InputDtype shA[size_k][tile_size_m], shB[size_k * tile_size_n];

    int offset = ((tile_c_idx_b * num_tiles_n + tile_c_idx_n) * tile_size_n + threadIdx.x) * num_tiles_m + tile_c_idx_m;

    InputDtype dq[8], scale_dq;
    TypeTraits<InputDtype>::e2m1_to_vec((uint32_t*) dq, x_e2m1_ptr[offset]);
    reinterpret_cast<uint16_t&>(scale_dq) = (uint16_t) x_e8m0_ptr[offset / (size_k / tile_size_m)] << 7;

    #pragma unroll
    for (int col = 0; col < tile_size_m; ++col) {
        shA[threadIdx.x][col] = dq[col] * scale_dq;
    }

    const InputDtype *shB_src = B + size_k * threadIdx.x;
    InputDtype *shB_dst = shB + size_k * threadIdx.x;

    #pragma unroll
    for (int col = 0; col < size_k; ++col) {
        shB_dst[col] = shB_src[col];
    }

    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tile_size_m, tile_size_n, tile_size_k, InputDtype, nvcuda::wmma::col_major> frag_a1, frag_a2;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tile_size_m, tile_size_n, tile_size_k, InputDtype, nvcuda::wmma::row_major> frag_b1, frag_b2;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tile_size_m, tile_size_n, tile_size_k, float> frag_c;

    nvcuda::wmma::fill_fragment(frag_c, 0.f);

    const InputDtype *tile_a1 = (InputDtype*) shA,
                     *tile_a2 = tile_a1 + tile_size_k * tile_size_m,
                     *tile_b1 = shB,
                     *tile_b2 = tile_b1 + tile_size_k * size_k;

    nvcuda::wmma::load_matrix_sync(frag_a1, tile_a1, tile_size_m);
    nvcuda::wmma::load_matrix_sync(frag_b1, tile_b1, size_k);

    nvcuda::wmma::mma_sync(frag_c, frag_a1, frag_b1, frag_c);

    nvcuda::wmma::load_matrix_sync(frag_a2, tile_a2, tile_size_m);
    nvcuda::wmma::load_matrix_sync(frag_b2, tile_b2, size_k);

    nvcuda::wmma::mma_sync(frag_c, frag_a2, frag_b2, frag_c);

    __shared__ float mat_c[tile_size_m][tile_size_n];
    nvcuda::wmma::store_matrix_sync(&mat_c[0][0], frag_c, tile_size_n, nvcuda::wmma::mem_row_major);

    if (threadIdx.x < tile_size_m) {
        float scale = 0.f;
        #pragma unroll
        for(int i = 0; i < tile_size_n; ++i) {
            float c_val = mat_c[threadIdx.x][i];
            scale = max(abs(c_val), scale);
        }
        scale /= alpha[0];
        reinterpret_cast<uint32_t&>(scale) = reinterpret_cast<uint32_t&>(scale) & 0x7f800000;
        xh_e8m0_ptr[((tile_c_idx_b * num_tiles_m + tile_c_idx_m) * tile_size_m + threadIdx.x) * num_tiles_n + tile_c_idx_n] = reinterpret_cast<uint32_t&>(scale) >> 23;
        #pragma unroll
        for(int i = 0; i < tile_size_n; ++i) {
            mat_c[threadIdx.x][i] *= 3.f / (scale * alpha[0]);
        }
    }

    __syncthreads();

    const int num_threads_per_row = blockDim.x / tile_size_m;
    xh_e2m1_ptr[(((tile_c_idx_b * num_tiles_m + tile_c_idx_m) * tile_size_m + threadIdx.x / num_threads_per_row) * num_tiles_n + tile_c_idx_n) * num_threads_per_row + threadIdx.x % num_threads_per_row] = fp32_vec_to_e2m1((float*) mat_c + tile_size_m * threadIdx.x);
}


int backward_t_bf16_cuda(const void* x_ptr,
                         const void* h_ptr,
                         void* xh_e2m1_ptr,
                         void* xh_e8m0_ptr,
                         const int size_m,
                         const int size_n,
                         const int size_b,
                         cudaStream_t stream)
{
    using InputDtype = __nv_bfloat16;
    dim3 grid((size_b * size_m * size_n / 32 + 8 - 1) / 8);
    dim3 block(32);
#if TARGET_CUDA_ARCH >= 100
    quantize_g32t_cuda_kernel<InputDtype><<<grid, block, 0, stream>>>(
        (InputDtype*) x_ptr,
        (InputDtype*) h_ptr,
        (uint32_t*) xh_e2m1_ptr,
        (uint8_t*) xh_e8m0_ptr,
        size_m,
        size_n,
        size_b
    );
#else
     TORCH_CHECK(false, "Unsupported CUDA arch");
#endif

    return 0;
}


int backward_qt_bf16_cuda(const void* x_e2m1_ptr,
                          const void* x_e8m0_ptr,
                          const void* h_ptr,
                          const void* alpha,
                          void* xh_e2m1_ptr,
                          void* xh_e8m0_ptr,
                          const int size_m,
                          const int size_n,
                          const int size_b,
                          cudaStream_t stream)
{
    using InputDtype = __nv_bfloat16;
    dim3 grid((size_b * size_m * size_n / 32 + 8 - 1) / 8);
    dim3 block(32);
#if TARGET_CUDA_ARCH >= 100
    quantize_g32qt_cuda_kernel<InputDtype><<<grid, block, 0, stream>>>(
        (uint32_t*) x_e2m1_ptr,
        (uint8_t*) x_e8m0_ptr,
        (InputDtype*) h_ptr,
        (float*) alpha,
        (uint32_t*) xh_e2m1_ptr,
        (uint8_t*) xh_e8m0_ptr,
        size_m,
        size_n,
        size_b
    );
#else
    TORCH_CHECK(false, "Unsupported CUDA arch");
#endif
    return 0;
}


#ifndef TILE
#define TILE 32
#endif

#define TILE_M 32
#define TILE_N 128

__device__ inline uint8_t encode_e8m0_shiftm8(float amax) {
    if (amax == 0.0f) return 127;
    __nv_bfloat16 b = __float2bfloat16_rn(amax);
    auto* br = reinterpret_cast<__nv_bfloat16_raw*>(&b);
    __nv_fp8_storage_t e8 = __nv_cvt_bfloat16raw_to_e8m0(*br, __NV_NOSAT, cudaRoundZero);
    return static_cast<uint8_t>(*reinterpret_cast<uint8_t*>(&e8) - 7);
}

__global__ void k_backward_bf16_square_double_mxfp8(
    const __nv_bfloat16* __restrict__ x_bf16, int ld_x,
    uint8_t*             __restrict__ y_fp8,  int ld_y,
    uint8_t*             __restrict__ row_scales, int ld_row,
    uint8_t*             __restrict__ col_scales, int ld_col,
    int m, int n)
{
    const int tile_i  = blockIdx.y;
    const int tile_j  = blockIdx.x;
    const int lane    = threadIdx.x%32;
    const int warp_id = threadIdx.x/32;

    const int gi  = tile_i * TILE_M + lane;
    const int gi0  = tile_i * TILE_M;

    const int gj0 = tile_j * TILE_N;

    if (gi >= m) return;

    __shared__ __nv_bfloat16 tile[TILE_M*(TILE_N+8)];

    const __nv_bfloat16* in_row = x_bf16 + gi * ld_x + gj0;

    const float4* x_bf16_f4 = reinterpret_cast<const float4*>(x_bf16 + gi0*ld_x + gj0 + warp_id*(TILE/4)*ld_x + (lane/16)*ld_x + (lane%16)*8);
    const float4* tile_f4   = reinterpret_cast<const float4*>(tile + warp_id*(TILE/4)*(TILE_N+8) + (lane/16)*(TILE_N+8) + (lane%16)*8);

    __nv_bfloat16 row_vals[TILE];

    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        *((float4*) tile_f4 + v*(TILE_N+8)/4) = *((float4*) x_bf16_f4 + v*ld_x/4); //2/8
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE/8; ++k) {
        *reinterpret_cast<float4*>(row_vals + k*8) = *reinterpret_cast<float4*>(tile + warp_id*(TILE) + lane*(TILE_N+8) + k*8);
    }

    float rmax = 0.f;
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        rmax = fmaxf(rmax, fabsf(__bfloat162float(row_vals[k])));
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFF, rmax, off));
    const float amax_tile = __shfl_sync(0xFFFFFFFF, rmax, 0);

    const uint8_t ebyte = encode_e8m0_shiftm8(amax_tile);

    __nv_fp8_storage_t e8; *reinterpret_cast<uint8_t*>(&e8) = ebyte;
    __nv_bfloat16_raw s_raw = __nv_cvt_e8m0_to_bf16raw(e8);
    const float qscale = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&s_raw));

    if (tile_j < n/TILE_N) {
        row_scales[(blockIdx.y*TILE_M + lane)*ld_row + (blockIdx.x)*(TILE_N/TILE) + warp_id] = ebyte;
    }

    const int go = gj0 + lane;
    if (go < n) {
        col_scales[(blockIdx.x*TILE_N*2+lane)*ld_col + (warp_id*TILE*2+lane)*ld_col + tile_i] = ebyte;
    }

    uint8_t out_bytes[TILE];

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const float q = __bfloat162float(row_vals[k]) / qscale;
        __nv_bfloat16 qb = __float2bfloat16_rn(q);
        __nv_fp8_storage_t f8 = __nv_cvt_bfloat16raw_to_fp8(*reinterpret_cast<__nv_bfloat16_raw*>(&qb),
                                                            __NV_SATFINITE, __NV_E4M3);
        out_bytes[k] = *reinterpret_cast<uint8_t*>(&f8);
    }

    const uint8_t* tile_u8 = reinterpret_cast<const uint8_t*>(tile);

    *((float4*) tile_u8 + warp_id*TILE/16 + lane*(TILE_N+16)/16)     = *((float4*) out_bytes + 0);
    *((float4*) tile_u8 + warp_id*TILE/16 + lane*(TILE_N+16)/16 + 1) = *((float4*) out_bytes + 1);

    __syncthreads();

    const float4* out_f4 =
        reinterpret_cast<const float4*>(y_fp8 + gi0*ld_y + gj0 + warp_id*(TILE/4)*ld_y + (lane/8)*ld_y + (lane%8)*16);
    const float4* out_tile_f4 =
        reinterpret_cast<const float4*>(tile_u8 + warp_id*(TILE/4)*(TILE_N+16) + (lane/8)*(TILE_N+16) + (lane%8)*16);

    *((float4*) out_f4)          = *((float4*) out_tile_f4) ;
    *((float4*) out_f4 + ld_y/4) = *((float4*) out_tile_f4 + (TILE_N+16)/4); //4/16
}

int backward_bf16_square_double_mxfp8_cuda(const void* x_bf16,
                                           const int m,
                                           const int n ,
                                           void* x_fp8,
                                           void* row_scales,
                                           void* column_scales,
                                           cudaStream_t stream)
{
    dim3 grid(n/TILE_N, m/TILE_M, 1);
    dim3 block(TILE_N, 1, 1);

    k_backward_bf16_square_double_mxfp8<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x_bf16, n,
        (uint8_t*) x_fp8, n,
        (uint8_t*) row_scales, n/TILE_M,
        (uint8_t*) column_scales, m/(TILE_M*2),
        m, n
    );

    return 0;
}

#define TILE_N_F4 256

__global__ void k_mxfp4_transpose_to_fp8_e4m3(
    const uint8_t* __restrict__ x_fp4_pairs,  int ld_fp4,
    const uint8_t* __restrict__ scales_e8m0,  int ld_scales,
    uint8_t*       __restrict__ y_fp8,        int ld_y,
    uint8_t*       __restrict__ out_e8m0,     int ld_out,
    int m, int n)
{
    const int tile_i  = blockIdx.y;
    const int tile_j  = blockIdx.x;
    const int lane    = threadIdx.x%32;
    const int warp_id = threadIdx.x/32;

    const int gi  = tile_i * TILE_M + lane;
    const int gi0 = tile_i * TILE_M;
    const int gj0 = tile_j * TILE_N_F4;

    const int go  = tile_j * TILE_N_F4 + lane;
    const int ho0 = tile_i * TILE_M;

    if (gi >= m || go >= n) return;

    __shared__ float tile_f32[TILE][TILE_N_F4];

    const uint8_t* row_fp4 = x_fp4_pairs + gi0*ld_fp4 + (gj0 >> 1) + warp_id*(TILE/8)*ld_fp4 + (lane/8)*ld_fp4 + (lane%8)*16; //TILE/8warps
    uint4 v16 = *reinterpret_cast<const uint4*>(row_fp4);

    uint8_t bytes16[16];
    *reinterpret_cast<uint4*>(bytes16) = v16;

    __nv_fp8_storage_t s8_in;
    *reinterpret_cast<uint8_t*>(&s8_in) = scales_e8m0[gi0*ld_scales + tile_j*(TILE_N_F4/TILE) + warp_id*(TILE/8)*ld_scales + (lane/8)*ld_scales + (lane%8)];
    __nv_bfloat16_raw s_in_raw = __nv_cvt_e8m0_to_bf16raw(s8_in);
    const float s_in = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&s_in_raw));

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        __nv_fp4x2_storage_t f4x2;
        *reinterpret_cast<uint8_t*>(&f4x2) = bytes16[i];
        __half2_raw h2r = __nv_cvt_fp4x2_to_halfraw2(f4x2, __NV_E2M1);
        const __half2 h2 = *reinterpret_cast<const __half2*>(&h2r);
        const float2 f2  = __half22float2(h2);

        tile_f32[threadIdx.x/8][2*i+0 + (threadIdx.x%8)*TILE] = __bfloat162float(__float2bfloat16_rn(f2.x * s_in));
        tile_f32[threadIdx.x/8][2*i+1 + (threadIdx.x%8)*TILE] = __bfloat162float(__float2bfloat16_rn(f2.y * s_in));
    }

    __syncthreads();

    //TODO: rf fst

    float amax = 0.f;
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        amax = fmaxf(amax, fabsf(tile_f32[k][lane + warp_id*TILE]));
    }
    const uint8_t ebyte_row = encode_e8m0_shiftm8(amax);

    out_e8m0[go*ld_out + tile_i + warp_id*TILE*ld_out] = ebyte_row;

    __nv_fp8_storage_t e8; *reinterpret_cast<uint8_t*>(&e8) = ebyte_row;
    __nv_bfloat16_raw s_row_raw = __nv_cvt_e8m0_to_bf16raw(e8);
    const float qscale = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&s_row_raw));

    uint8_t out_bytes[32];
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const float q = tile_f32[k][lane + warp_id*TILE] / qscale;
        __nv_bfloat16 qb = __float2bfloat16_rn(q);
        __nv_fp8_storage_t f8 =
            __nv_cvt_bfloat16raw_to_fp8(*reinterpret_cast<__nv_bfloat16_raw*>(&qb),
                                        __NV_SATFINITE, __NV_E4M3);
        out_bytes[k] = *reinterpret_cast<uint8_t*>(&f8);
    }

    //TODO: smem fst

    uint8_t* out_row = y_fp8 + go*ld_y + ho0 + warp_id*TILE*ld_y;
    if (ho0 + 31 < m) {
        *reinterpret_cast<uint4*>(out_row +  0) = *reinterpret_cast<const uint4*>(out_bytes +  0);
        *reinterpret_cast<uint4*>(out_row + 16) = *reinterpret_cast<const uint4*>(out_bytes + 16);
    } else {
        #pragma unroll
        for (int k = 0; k < TILE; ++k) if (ho0 + k < m) out_row[k] = out_bytes[k];
    }
}

int mxfp4_transpose_mxfp8_cuda(
    const void* x_fp4_pairs,
    const void* scales_e8m0,
    int m, int n,
    void* x_fp8,
    void* shared_exps,
    cudaStream_t stream)
{
    dim3 grid(n/TILE_N_F4, m/TILE, 1);
    dim3 block(TILE_N_F4, 1, 1);

    k_mxfp4_transpose_to_fp8_e4m3<<<grid, block, 0, stream>>>(
        (const uint8_t*) x_fp4_pairs, n/2,
        (const uint8_t*) scales_e8m0, n/32,
        (uint8_t*) x_fp8, m,
        (uint8_t*) shared_exps, m/32,
        m, n
    );

    return 0;
}
