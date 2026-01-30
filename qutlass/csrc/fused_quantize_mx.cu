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

#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#ifndef QUTLASS_DISABLE_PYBIND
#include <torch/extension.h>
#endif

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

template <typename ShapeMMAThreadBlock, typename ShapeMMAWarp, typename InstructionShape, bool Quest=true, int RotationSize=32>
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
        RotationSize
    >;

template <typename Gemm>
struct GemmRunner {
  uint64_t seed;

  GemmRunner() { }

  bool run(
    torch::Tensor &out,
    torch::Tensor &out_sf,
    torch::Tensor const&x,
    torch::Tensor const&y,
    int32_t M, int32_t N, int32_t K,
    torch::Device device)
  {

    using GemmCoord = cutlass::gemm::GemmCoord;
    Gemm gemmOp;

    typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M),
       static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {(cutlass::bfloat16_t *)x.data_ptr(), K},
      {(cutlass::bfloat16_t *)y.data_ptr(), N},
      {(cutlass::float_e2m1_t *)out.data_ptr(), N},
      {(cutlass::float_e2m1_t *)out.data_ptr(), N},
      {(cutlass::float_ue8m0_t *)out_sf.data_ptr(), M},
        cutlass::bfloat16_t(0) //TODO (later): float
    };

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

    CUTLASS_CHECK(gemmOp.initialize(arguments, nullptr, stream));

    CUTLASS_CHECK(gemmOp(arguments, nullptr, stream));

    return true;
  }

};


void fusedQuantizeMxQuest_host(torch::Tensor& D,
                               torch::Tensor& D_sf,
                               torch::Tensor const& A,
                               torch::Tensor const& B)
{
  int32_t M = A.numel() / 32;
  int32_t N = B.size(1);
  int32_t K = 32;

  using TileShape = typename cutlass::gemm::GemmShape<128, 32, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 32, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  GemmRunner<Gemm_<TileShape, WarpShape, MmaShape, true, 32>> runGemm;
  bool result = runGemm.run(D, D_sf, A, B, M, N, K, A.device());
}

void fusedQuantizeMxAbsMax_host(torch::Tensor& D,
                                torch::Tensor& D_sf,
                                torch::Tensor const& A,
                                torch::Tensor const& B)
{
  int32_t M = A.numel() / 32;
  int32_t N = B.size(1);
  int32_t K = 32;

  using TileShape = typename cutlass::gemm::GemmShape<128, 32, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 32, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  GemmRunner<Gemm_<TileShape, WarpShape, MmaShape, false, 32>> runGemm;
  bool result = runGemm.run(D, D_sf, A, B, M, N, K, A.device());
}

void fusedQuantizeMxQuestHad64_host(torch::Tensor& D,
                               torch::Tensor& D_sf,
                               torch::Tensor const& A,
                               torch::Tensor const& B)
{
  int32_t M = A.numel() / 64;
  int32_t N = B.size(1);
  int32_t K = 64;

  using TileShape = typename cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  GemmRunner<Gemm_<TileShape, WarpShape, MmaShape, true, 64>> runGemm;
  bool result = runGemm.run(D, D_sf, A, B, M, N, K, A.device());
}

void fusedQuantizeMxAbsMaxHad64_host(torch::Tensor& D,
                                torch::Tensor& D_sf,
                                torch::Tensor const& A,
                                torch::Tensor const& B)
{
  int32_t M = A.numel() / 64;
  int32_t N = B.size(1);
  int32_t K = 64;

  using TileShape = typename cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  GemmRunner<Gemm_<TileShape, WarpShape, MmaShape, false, 64>> runGemm;
  bool result = runGemm.run(D, D_sf, A, B, M, N, K, A.device());
}

void fusedQuantizeMxQuestHad128_host(torch::Tensor& D,
                               torch::Tensor& D_sf,
                               torch::Tensor const& A,
                               torch::Tensor const& B)
{
  int32_t M = A.numel() / 128;
  int32_t N = B.size(1);
  int32_t K = 128;

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 128, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  GemmRunner<Gemm_<TileShape, WarpShape, MmaShape, true, 128>> runGemm;
  bool result = runGemm.run(D, D_sf, A, B, M, N, K, A.device());
}

void fusedQuantizeMxAbsMaxHad128_host(torch::Tensor& D,
                                torch::Tensor& D_sf,
                                torch::Tensor const& A,
                                torch::Tensor const& B)
{
  int32_t M = A.numel() / 128;
  int32_t N = B.size(1);
  int32_t K = 128;

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 128, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  GemmRunner<Gemm_<TileShape, WarpShape, MmaShape, false, 128>> runGemm;
  bool result = runGemm.run(D, D_sf, A, B, M, N, K, A.device());
}

void fusedQuantizeMxQuestHad256_host(torch::Tensor& D,
                               torch::Tensor& D_sf,
                               torch::Tensor const& A,
                               torch::Tensor const& B)
{
  int32_t M = A.numel() / 256;
  int32_t N = B.size(1);
  int32_t K = 256;

  using TileShape = typename cutlass::gemm::GemmShape<128, 256, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 256, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  GemmRunner<Gemm_<TileShape, WarpShape, MmaShape, true, 256>> runGemm;
  bool result = runGemm.run(D, D_sf, A, B, M, N, K, A.device());
}

void fusedQuantizeMxAbsMaxHad256_host(torch::Tensor& D,
                                torch::Tensor& D_sf,
                                torch::Tensor const& A,
                                torch::Tensor const& B)
{
  int32_t M = A.numel() / 256;
  int32_t N = B.size(1);
  int32_t K = 256;

  using TileShape = typename cutlass::gemm::GemmShape<128, 256, 32>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 256, 32>;
  using MmaShape  = typename cutlass::gemm::GemmShape<16, 8, 16>;

  GemmRunner<Gemm_<TileShape, WarpShape, MmaShape, false, 256>> runGemm;
  bool result = runGemm.run(D, D_sf, A, B, M, N, K, A.device());
}

} // namespace QUTLASS