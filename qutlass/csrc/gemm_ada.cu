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

#include <stddef.h>

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

#include "cutlass_extensions/gemm/device/mx_gemm.h"

#include <gemm.h>

template <typename TileShape, typename WarpShape, int kStages>
void qutlass_matmul_mxf4_v1(torch::Tensor const&input,
                            torch::Tensor const&weight,
                            torch::Tensor const&input_sf,
                            torch::Tensor const&weight_sf,
                            torch::Tensor &out,
                            torch::Tensor const& alpha,
                            torch::Device device)
{
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1)*2;

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = cutlass::float_e2m1_t;
  using ElementInputB = cutlass::float_e2m1_t;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::MxGemm<
      ElementInputA,
      cutlass::layout::RowMajor,
      ElementInputB,
      cutlass::layout::ColumnMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TileShape,
      WarpShape,
      cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput,
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,
          ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      kStages>;

  auto input_size  = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      static_cast<ElementInputA*>(input.data_ptr()),
      LayoutInputA::packed(input_size));

  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      static_cast<ElementInputB*>(weight.data_ptr()),
      LayoutInputB::packed(weight_size));

  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      static_cast<ElementOutput*>(out.data_ptr()),
      LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size,
      input_ref,
      reinterpret_cast<const cutlass::float_ue8m0_t*>(input_sf.data_ptr()),
      weight_ref,
      reinterpret_cast<const cutlass::float_ue8m0_t*>(weight_sf.data_ptr()),
      out_ref,
      out_ref,
      {},
      1
  };
  arguments.epilogue.alpha_ptr = static_cast<ElementAccumulator const*>(alpha.data_ptr());

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get(), stream));

  CUTLASS_CHECK(gemm_op(stream));
}

void matmul_host_ada_mxf4_bf16_tn(torch::Tensor const&input,
                                  torch::Tensor const&weight,
                                  torch::Tensor const&input_sf,
                                  torch::Tensor const&weight_sf,
                                  torch::Tensor &out,
                                  torch::Tensor const& alpha)
{
  using TileShape = typename cutlass::gemm::GemmShape<16, 16, 256>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 16, 256>;
  static const int kStages = 5;

#if TARGET_CUDA_ARCH == 120
  qutlass_matmul_mxf4_v1<TileShape, WarpShape, kStages>(input, weight, input_sf, weight_sf, out, alpha, input.device());
#else
    TORCH_CHECK(false, "matmul_ada_mxf4_bf16_tn was optimized for sm120. For other architectures, please use matmul_mxf4_bf16_tn instead");
#endif
}