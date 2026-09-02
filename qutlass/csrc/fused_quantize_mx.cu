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

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "fused_quantize_host.h"
#include "cutlass_extensions/gemm/device/gemm_quant.h"

namespace QUTLASS {

using ElementInputA     = cutlass::bfloat16_t;
using ElementInputB     = cutlass::bfloat16_t;
using ElementGemmOutput = cutlass::bfloat16_t; //TODO (later):
using ElementOutput     = cutlass::float_e2m1_t;
using ElementAuxOutput  = ElementOutput;

using ElementAccumulator     = float;
using ElementComputeEpilogue = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

template <bool SfSwizzled>
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationQuantMx<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementGemmOutput>::value,
    ElementAccumulator,
    ElementGemmOutput,
    cutlass::epilogue::thread::MyScaleType::Quantize,
    cutlass::FloatRoundStyle::round_to_nearest,
    ElementGemmOutput,
    SfSwizzled>;

template <typename ShapeMMAThreadBlock, typename ShapeMMAWarp,
          typename InstructionShape, bool Quest = true, int RotationSize = 32,
          bool SfSwizzled = true>
using Gemm_ =
    cutlass::gemm::device::GemmQuantMx<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementGemmOutput, LayoutOutput,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ShapeMMAThreadBlock,
        ShapeMMAWarp,
        InstructionShape,
        Quest,
        RotationSize,
        EpilogueOutputOp<SfSwizzled>
    >;

template <typename Gemm>
struct GemmRunner {
  uint64_t seed;

  GemmRunner() { }

  bool run(
    torch::stable::Tensor &out,
    torch::stable::Tensor &out_sf,
    torch::stable::Tensor const& x,
    torch::stable::Tensor const& y,
    int32_t M, int32_t N, int32_t K,
    torch::stable::Device device,
    float* global_scale = nullptr,
    int n_col_blocks = 1,
    int logical_sf_cols = 1)
  {

    using GemmCoord = cutlass::gemm::GemmCoord;
    Gemm gemmOp;

    typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M),
       static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {static_cast<const cutlass::bfloat16_t*>(x.const_data_ptr()), K},
      {static_cast<const cutlass::bfloat16_t*>(y.const_data_ptr()), N},
      {static_cast<cutlass::float_e2m1_t*>(out.mutable_data_ptr()), N},
      {static_cast<cutlass::float_e2m1_t*>(out.mutable_data_ptr()), N},
      {static_cast<cutlass::float_ue8m0_t*>(out_sf.mutable_data_ptr()), M},
        global_scale,
        n_col_blocks,
        logical_sf_cols,
        (int)out_sf.numel(),
        cutlass::bfloat16_t(0) //TODO (later): float
    };

    const torch::stable::accelerator::DeviceGuard device_guard(x.get_device_index());
    cudaStream_t stream = get_current_cuda_stream(device.index());


    CUTLASS_CHECK(gemmOp.initialize(arguments, nullptr, stream));

    CUTLASS_CHECK(gemmOp(arguments, nullptr, stream));

    return true;
  }

};

template <bool SfSwizzled, typename TileShape, typename WarpShape,
          typename MmaShape, bool Quest, int RotationSize>
void run_quantize_mx(torch::stable::Tensor& D,
                     torch::stable::Tensor& D_sf,
                     torch::stable::Tensor const& A,
                     torch::stable::Tensor const& B,
                     int32_t M, int32_t N, int32_t K,
                     float* global_scale) {
  GemmRunner<Gemm_<TileShape, WarpShape, MmaShape, Quest, RotationSize,
                   SfSwizzled>> runner;
  runner.run(D, D_sf, A, B, M, N, K, A.device(), global_scale,
             D_sf.size(1) / 4, A.size(-1) / 32);
}

template <typename TileShape, typename WarpShape, typename MmaShape,
          bool Quest, int RotationSize>
void dispatch_quantize_mx(torch::stable::Tensor& D,
                          torch::stable::Tensor& D_sf,
                          torch::stable::Tensor const& A,
                          torch::stable::Tensor const& B,
                          int32_t M, int32_t N, int32_t K,
                          float* global_scale,
                          bool is_sf_swizzled_layout) {
  if (is_sf_swizzled_layout) {
    run_quantize_mx<true, TileShape, WarpShape, MmaShape, Quest, RotationSize>(
        D, D_sf, A, B, M, N, K, global_scale);
  } else {
    run_quantize_mx<false, TileShape, WarpShape, MmaShape, Quest, RotationSize>(
        D, D_sf, A, B, M, N, K, global_scale);
  }
}


void fusedQuantizeMxQuest_host(torch::stable::Tensor& D,
                               torch::stable::Tensor& D_sf,
                               torch::stable::Tensor const& A,
                               torch::stable::Tensor const& B,
                               bool is_sf_swizzled_layout)
{
  int32_t M = A.numel() / 32;
  int32_t N = B.size(1);
  int32_t K = 32;

  using TileShape = typename cutlass::gemm::GemmShape<128, 32, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 32, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  dispatch_quantize_mx<TileShape, WarpShape, MmaShape, true, 32>(
      D, D_sf, A, B, M, N, K, nullptr, is_sf_swizzled_layout);
}

void fusedQuantizeMxAbsMax_host(torch::stable::Tensor& D,
                                torch::stable::Tensor& D_sf,
                                torch::stable::Tensor const& A,
                                torch::stable::Tensor const& B,
                                torch::stable::Tensor const& global_scale,
                                bool is_sf_swizzled_layout)
{
  int32_t M = A.numel() / 32;
  int32_t N = B.size(1);
  int32_t K = 32;

  using TileShape = typename cutlass::gemm::GemmShape<128, 32, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 32, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  dispatch_quantize_mx<TileShape, WarpShape, MmaShape, false, 32>(
      D, D_sf, A, B, M, N, K,
      static_cast<float*>(const_cast<void*>(global_scale.const_data_ptr())),
      is_sf_swizzled_layout);
}

void fusedQuantizeMxQuestHad64_host(torch::stable::Tensor& D,
                               torch::stable::Tensor& D_sf,
                               torch::stable::Tensor const& A,
                               torch::stable::Tensor const& B,
                               bool is_sf_swizzled_layout)
{
  int32_t M = A.numel() / 64;
  int32_t N = B.size(1);
  int32_t K = 64;

  using TileShape = typename cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  dispatch_quantize_mx<TileShape, WarpShape, MmaShape, true, 64>(
      D, D_sf, A, B, M, N, K, nullptr, is_sf_swizzled_layout);
}

void fusedQuantizeMxAbsMaxHad64_host(torch::stable::Tensor& D,
                                torch::stable::Tensor& D_sf,
                                torch::stable::Tensor const& A,
                                torch::stable::Tensor const& B,
                                     torch::stable::Tensor const& global_scale,
                                     bool is_sf_swizzled_layout)
{
  int32_t M = A.numel() / 64;
  int32_t N = B.size(1);
  int32_t K = 64;

  using TileShape = typename cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  dispatch_quantize_mx<TileShape, WarpShape, MmaShape, false, 64>(
      D, D_sf, A, B, M, N, K,
      static_cast<float*>(const_cast<void*>(global_scale.const_data_ptr())),
      is_sf_swizzled_layout);
}

void fusedQuantizeMxQuestHad128_host(torch::stable::Tensor& D,
                               torch::stable::Tensor& D_sf,
                               torch::stable::Tensor const& A,
                                     torch::stable::Tensor const& B,
                                     bool is_sf_swizzled_layout)
{
  int32_t M = A.numel() / 128;
  int32_t N = B.size(1);
  int32_t K = 128;

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 128, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  dispatch_quantize_mx<TileShape, WarpShape, MmaShape, true, 128>(
      D, D_sf, A, B, M, N, K, nullptr, is_sf_swizzled_layout);
}

void fusedQuantizeMxAbsMaxHad128_host(torch::stable::Tensor& D,
                                torch::stable::Tensor& D_sf,
                                torch::stable::Tensor const& A,
                                torch::stable::Tensor const& B,
                                      torch::stable::Tensor const& global_scale,
                                      bool is_sf_swizzled_layout)
{
  int32_t M = A.numel() / 128;
  int32_t N = B.size(1);
  int32_t K = 128;

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 128, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  dispatch_quantize_mx<TileShape, WarpShape, MmaShape, false, 128>(
      D, D_sf, A, B, M, N, K,
      static_cast<float*>(const_cast<void*>(global_scale.const_data_ptr())),
      is_sf_swizzled_layout);
}

} // namespace QUTLASS
