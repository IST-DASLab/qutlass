#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#ifndef QUTLASS_DISABLE_PYBIND
#include <torch/extension.h>
#endif

#include <backward_host.h>

// Generic template for type-specific conversion
template <typename T>
struct TypeTraits;

// Specialization for __half
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

// Specialization for __nv_bfloat16
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
#if TARGET_CUDA_ARCH == 120
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
#if TARGET_CUDA_ARCH == 120
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
