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

template <typename ShapeMMAThreadBlock, typename ShapeMMAWarp, typename InstructionShape>
using Gemm_ =
    cutlass::gemm::device::GemmQuantMxMask<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementGemmOutput, LayoutOutput,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ShapeMMAThreadBlock,
        ShapeMMAWarp,
        InstructionShape
    >;

template <typename Gemm>
struct GemmRunner {
  uint64_t seed;

  GemmRunner() { }

  bool run(
    torch::stable::Tensor &out,
    torch::stable::Tensor &out_sf,
    torch::stable::Tensor &out_mask,
    torch::stable::Tensor const& x,
    torch::stable::Tensor const& y,
    int32_t M, int32_t N, int32_t K,
    torch::stable::Device device)
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
      {static_cast<uint8_t*>(out_mask.mutable_data_ptr()), M}, //FIXME: bfloat16_t
        cutlass::bfloat16_t(0) //TODO (later): float
    };

    const torch::stable::accelerator::DeviceGuard device_guard(x.get_device_index());
    cudaStream_t stream = get_current_cuda_stream(device.index());

    CUTLASS_CHECK(gemmOp.initialize(arguments, nullptr, stream));

    CUTLASS_CHECK(gemmOp(arguments, nullptr, stream));

    return true;
  }

};

void fusedQuantizeMxQuestWithMask_host(torch::stable::Tensor& D,
                                       torch::stable::Tensor& D_sf,
                                       torch::stable::Tensor& D_mask,
                                       torch::stable::Tensor const& A,
                                       torch::stable::Tensor const& B)
{
  int32_t M = A.numel() / 32;
  int32_t N = B.size(1);
  int32_t K = 32;

  using TileShape = typename cutlass::gemm::GemmShape<128, 32, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 32, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  GemmRunner<Gemm_<TileShape, WarpShape, MmaShape>> runGemm;
  bool result = runGemm.run(D, D_sf, D_mask, A, B, M, N, K, A.device());
}

} // namespace QUTLASS