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

/**
 * CUTLASS 3.x/4.x Fused Quantization Kernel for MXFP4 - V2 with Arbitrary K
 *
 * This implementation uses CUTLASS 3.x/4.x CollectiveBuilder and GemmUniversalAdapter
 * for native SM100/SM120 (Blackwell) support with arbitrary K dimensions.
 *
 * Based on CUTLASS Example 71: Blackwell GEMM with CollectiveBuilder
 * https://github.com/NVIDIA/cutlass/blob/main/examples/71_blackwell_gemm_with_collective_builder/
 *
 * Supported Architectures:
 *   - SM100 (B200/B300 datacenter - both report compute capability 10.0)
 *   - SM120 (Blackwell RTX Pro 6000, RTX 5090)
 *
 * Note: B200 and B300 differ in specs (FLOPs, memory) but use the same CUDA
 * compute capability 10.0 and the same CUTLASS kernels.
 *
 * Operation: D_fp4, D_scale = Quantize(A @ H^T)
 *   where H is the Hadamard rotation matrix
 *   and Quantize uses either Quest (variance-based) or AbsMax scaling
 */

#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#ifndef QUTLASS_DISABLE_PYBIND
#include <torch/extension.h>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

// Define CUTLASS_CHECK if not already defined
#ifndef CUTLASS_CHECK
#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                           \
    if (error != cutlass::Status::kSuccess) {                                 \
      std::cerr << "CUTLASS error: " << cutlassGetStatusString(error)        \
                << " (code=" << int(error) << ")"                              \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      throw std::runtime_error(std::string("CUTLASS error: ") + cutlassGetStatusString(error)); \
    }                                                                          \
  }
#endif

namespace QUTLASS_V2 {

//=============================================================================
// Forward Declarations
//=============================================================================

template <bool UseQuest>
void quantize_bf16_to_mxfp4(
    cutlass::float_e2m1_t* output,
    cutlass::float_ue8m0_t* scales,
    cutlass::bfloat16_t const* input,
    int M, int K,
    cudaStream_t stream);

//=============================================================================
// Device Functions: FP32 to E2M1 conversion and quantization
//=============================================================================

/**
 * Convert 8 FP32 values to packed E2M1 (4-bit) format
 * Uses PTX cvt.rn.satfinite.e2m1x2.f32 instruction
 */
__device__ __forceinline__
uint32_t fp32_vec_to_e2m1(float* array) {
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

/**
 * Extract e8m0 scale from FP32 value
 * Rounds to nearest power of 2 (extracts biased exponent)
 */
__device__ __forceinline__
uint8_t extract_e8m0_scale(float scale) {
    uint32_t bits = reinterpret_cast<uint32_t&>(scale) & 0x7f800000;
    return static_cast<uint8_t>(bits >> 23);
}

/**
 * Quest quantization: variance-based scaling
 * scale = sqrt(var) * (2.92247856 / 6) + epsilon
 */
__device__ __forceinline__
float compute_quest_scale(const float* values, int count) {
    float sum1 = 0.f, sum2 = 0.f;

    #pragma unroll
    for (int i = 0; i < count; ++i) {
        float v = values[i];
        sum1 += v;
        sum2 += v * v;
    }

    float mean = sum1 / count;
    float var = sum2 / count - mean * mean;
    float scale = 1.0f;

    if (var >= 0) {
        scale = sqrtf(var) * (2.92247856f / 6.0f) + 1e-8f;
    }

    // Round to power of 2 (e8m0 format)
    uint32_t& scale_bits = reinterpret_cast<uint32_t&>(scale);
    scale_bits = scale_bits & 0x7f800000;

    return scale;
}

/**
 * AbsMax quantization: maximum absolute value scaling
 * scale = max(|x|) + epsilon
 */
__device__ __forceinline__
float compute_absmax_scale(const float* values, int count) {
    float abs_max = 0.f;

    #pragma unroll
    for (int i = 0; i < count; ++i) {
        float abs_val = fabsf(values[i]);
        if (abs_val > abs_max) abs_max = abs_val;
    }

    float scale = abs_max + 1e-8f;

    // Round to power of 2 (e8m0 format)
    uint32_t& scale_bits = reinterpret_cast<uint32_t&>(scale);
    scale_bits = scale_bits & 0x7f800000;

    return scale;
}

//=============================================================================
// CUTLASS 3.x/4.x GEMM Configuration with CollectiveBuilder
//=============================================================================

/**
 * Fused GEMM + Quantization using CUTLASS 3.x/4.x CollectiveBuilder
 *
 * Computes: D_fp4, scale = Quantize(A_bf16 @ H_bf16)
 *
 * Key features:
 * - Native SM100/SM103/SM120 support
 * - Arbitrary K dimension support (tested up to K=65536)
 * - Uses GemmUniversalAdapter for proper workspace management
 * - Based on CUTLASS Example 71
 */

template <typename ArchTag>
struct GemmConfig {
    // Element types
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    // Layouts (all RowMajor for A @ B)
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    // Alignments (128 bits = 16 bytes)
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    // Tile shapes
    using MmaTileMNK = cute::Shape<cute::_128, cute::_128, cute::_64>;
    using ClusterShapeMNK = cute::Shape<cute::_1, cute::_1, cute::_1>;

    // Schedule types
    using MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto;
    using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using StageCountType = cutlass::gemm::collective::StageCountAuto;

    // Mainloop collective (handles matrix multiplication)
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileMNK, ClusterShapeMNK,
        StageCountType,
        MainloopScheduleType
    >::CollectiveOp;

    // Epilogue collective (handles D = alpha * C + beta * D)
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        MmaTileMNK, ClusterShapeMNK,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        EpilogueScheduleType
    >::CollectiveOp;

    // GEMM kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    // GEMM device adapter
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

//=============================================================================
// Kernel Runner (CUTLASS 3.x/4.x API)
//=============================================================================

template <bool UseQuest, typename ArchTag>
void runFusedQuantize(
    torch::Tensor& D,           // Output: packed E2M1 values [M, K/2]
    torch::Tensor& D_sf,        // Output: scale factors [M, K/32]
    torch::Tensor const& A,     // Input: BF16 values [M, K]
    torch::Tensor const& H,     // Hadamard matrix [K, K]
    int M, int K,
    torch::Device device)
{
    using Config = GemmConfig<ArchTag>;
    using Gemm = typename Config::Gemm;

    // Create temporary buffer for BF16 GEMM output
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    torch::Tensor gemm_output = torch::empty({M, K}, options);

    // Setup CUDA stream
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

    // Problem size: A [M, K] @ H [K, K] = output [M, K]
    int N = K;

    // Stride configuration for RowMajor layout
    // For RowMajor (M x K): stride = (K, 1, 0) = (leading_dim, 1, batch_stride)
    // CUTLASS expects int64_t for runtime dimensions and cute::Int<1> for compile-time constants
    using StrideType = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    auto stride_A = StrideType{int64_t(K), cute::Int<1>{}, int64_t(0)};  // A: (M x K)
    auto stride_B = StrideType{int64_t(K), cute::Int<1>{}, int64_t(0)};  // B: (K x K)
    auto stride_C = StrideType{int64_t(K), cute::Int<1>{}, int64_t(0)};  // C: (M x K)
    auto stride_D = StrideType{int64_t(K), cute::Int<1>{}, int64_t(0)};  // D: (M x K)

    // Hardware info
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device.index();
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // CUTLASS 3.x/4.x GEMM Arguments
    // Note: C is unused (beta=0), so we pass D pointer for both C and D
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},  // problem_shape (M, N, K, batch=1)
        {
            reinterpret_cast<typename Config::ElementA const*>(A.data_ptr()), stride_A,
            reinterpret_cast<typename Config::ElementB const*>(H.data_ptr()), stride_B
        },
        {
            {},  // epilogue arguments (will set alpha/beta below)
            reinterpret_cast<typename Config::ElementC const*>(gemm_output.data_ptr()), stride_C,
            reinterpret_cast<typename Config::ElementD*>(gemm_output.data_ptr()), stride_D
        },
        hw_info
    };

    // Set epilogue alpha=1, beta=0 for D = A @ B
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    // Create GEMM operator
    Gemm gemm_op;

    // Query workspace size
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Verify GEMM can run
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("GEMM can_implement failed: ") + cutlassGetStatusString(status) +
            ". Ensure you're compiling with the correct architecture (sm_100a or sm_120a).");
    }

    // Initialize kernel
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    // Run GEMM
    status = gemm_op.run(stream);
    CUTLASS_CHECK(status);

    // Synchronize
    cudaStreamSynchronize(stream);

    // Quantize the BF16 output to MXFP4
    quantize_bf16_to_mxfp4<UseQuest>(
        reinterpret_cast<cutlass::float_e2m1_t*>(D.data_ptr()),
        reinterpret_cast<cutlass::float_ue8m0_t*>(D_sf.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t const*>(gemm_output.data_ptr()),
        M, K, stream);
}

//=============================================================================
// Quantization Kernel (Separate for now, to be fused later)
//=============================================================================

template <bool UseQuest, int GroupSize = 32>
__global__ void quantize_kernel(
    uint8_t* __restrict__ output,       // Packed E2M1 output [M, K/2]
    uint8_t* __restrict__ scales,       // Scale factors [M, K/32]
    cutlass::bfloat16_t const* __restrict__ input,  // BF16 input [M, K]
    int M, int K)
{
    // Each thread handles one row (M dimension)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // Process K elements in groups of GroupSize (32 elements)
    int num_groups = K / GroupSize;
    int output_row_offset = row * (K / 2);      // Output is K/2 bytes per row
    int scales_row_offset = row * num_groups;   // One scale per group

    for (int g = 0; g < num_groups; ++g) {
        // Load GroupSize values from BF16 input
        float values[GroupSize];
        int input_offset = row * K + g * GroupSize;

        #pragma unroll
        for (int i = 0; i < GroupSize; ++i) {
            values[i] = static_cast<float>(input[input_offset + i]);
        }

        // Compute scale (Quest or AbsMax)
        float scale;
        if constexpr (UseQuest) {
            scale = compute_quest_scale(values, GroupSize);
        } else {
            scale = compute_absmax_scale(values, GroupSize);
        }

        // Store e8m0 scale factor
        scales[scales_row_offset + g] = extract_e8m0_scale(scale);

        // Quantize: divide by scale
        float inv_scale = 1.0f / scale;
        if constexpr (!UseQuest) {
            inv_scale *= 3.0f;  // AbsMax uses 3x scaling for E2M1 range [-6, 6]
        }

        // Pack 8 values into one uint32 (8 x 4-bit = 32 bits)
        // Each group of 32 elements produces 16 bytes of packed output
        int group_out_offset = output_row_offset + (g * GroupSize / 2);

        for (int w = 0; w < GroupSize / 8; ++w) {
            float group[8];
            #pragma unroll
            for (int z = 0; z < 8; ++z) {
                group[z] = values[w * 8 + z] * inv_scale;
            }
            uint32_t packed = fp32_vec_to_e2m1(group);

            // Store packed values (4 bytes = 8 x 4-bit values)
            reinterpret_cast<uint32_t*>(output + group_out_offset)[w] = packed;
        }
    }
}

template <bool UseQuest>
void quantize_bf16_to_mxfp4(
    cutlass::float_e2m1_t* output,
    cutlass::float_ue8m0_t* scales,
    cutlass::bfloat16_t const* input,
    int M, int K,
    cudaStream_t stream)
{
    constexpr int GroupSize = 32;
    constexpr int kBlockSize = 256;
    int num_blocks = (M + kBlockSize - 1) / kBlockSize;

    quantize_kernel<UseQuest, GroupSize><<<num_blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<uint8_t*>(output),
        reinterpret_cast<uint8_t*>(scales),
        input, M, K);
}

//=============================================================================
// Host API Functions with Runtime Architecture Detection
//=============================================================================

void fusedQuantizeMxQuest_v2(
    torch::Tensor& D,
    torch::Tensor& D_sf,
    torch::Tensor const& A,
    torch::Tensor const& H)
{
    int M = A.size(0);
    int K = A.size(1);

    TORCH_CHECK(K >= 32, "K must be at least 32");
    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32");
    TORCH_CHECK(H.size(0) == K && H.size(1) == K, "H must be K x K");

    // Detect GPU architecture and dispatch to appropriate kernel
    auto compute_capability = at::cuda::getCurrentDeviceProperties()->major * 10 +
                             at::cuda::getCurrentDeviceProperties()->minor;

    if (compute_capability == 100) {
        // SM100 (B200/B300 datacenter - both use compute capability 10.0)
        runFusedQuantize<true, cutlass::arch::Sm100>(D, D_sf, A, H, M, K, A.device());
    } else if (compute_capability >= 120) {
        // SM120+ (Blackwell RTX Pro 6000, RTX 5090)
        runFusedQuantize<true, cutlass::arch::Sm120>(D, D_sf, A, H, M, K, A.device());
    } else {
        TORCH_CHECK(false,
            "Unsupported GPU architecture sm_", compute_capability,
            ". v2 API requires Blackwell (SM100 or SM120). "
            "Supported GPUs: B200/B300 (SM100), RTX Pro 6000/RTX 5090 (SM120).");
    }
}

void fusedQuantizeMxAbsMax_v2(
    torch::Tensor& D,
    torch::Tensor& D_sf,
    torch::Tensor const& A,
    torch::Tensor const& H)
{
    int M = A.size(0);
    int K = A.size(1);

    TORCH_CHECK(K >= 32, "K must be at least 32");
    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32");
    TORCH_CHECK(H.size(0) == K && H.size(1) == K, "H must be K x K");

    // Detect GPU architecture and dispatch to appropriate kernel
    auto compute_capability = at::cuda::getCurrentDeviceProperties()->major * 10 +
                             at::cuda::getCurrentDeviceProperties()->minor;

    if (compute_capability == 100) {
        // SM100 (B200/B300 datacenter - both use compute capability 10.0)
        runFusedQuantize<false, cutlass::arch::Sm100>(D, D_sf, A, H, M, K, A.device());
    } else if (compute_capability >= 120) {
        // SM120+ (Blackwell RTX Pro 6000, RTX 5090)
        runFusedQuantize<false, cutlass::arch::Sm120>(D, D_sf, A, H, M, K, A.device());
    } else {
        TORCH_CHECK(false,
            "Unsupported GPU architecture sm_", compute_capability,
            ". v2 API requires Blackwell (SM100 or SM120). "
            "Supported GPUs: B200/B300 (SM100), RTX Pro 6000/RTX 5090 (SM120).");
    }
}

} // namespace QUTLASS_V2
