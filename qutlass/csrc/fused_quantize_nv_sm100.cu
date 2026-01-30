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

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "cutlass_extensions/epilogue/collective/collective_builder.hpp"
#include "fused_quantize_host.h"

using namespace cute;

namespace QUTLASS {

using         ElementA    = cutlass::bfloat16_t;
using         LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;

using         ElementB    = cutlass::bfloat16_t;
using         LayoutB     = cutlass::layout::RowMajor;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;

using         ElementC    = void;
using         LayoutC     = cutlass::layout::RowMajor;
constexpr int AlignmentC  = 1;

using         ElementD    = cutlass::float_e2m1_t;
using         LayoutD     = cutlass::layout::RowMajor;
using         LayoutSFDTag = LayoutD;
using         ElementSFD  = cutlass::float_ue4m3_t;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator  = float;
using ElementCompute      = float;
using ArchTag             = cutlass::arch::Sm100;
using OperatorClass       = cutlass::arch::OpClassTensorOp;

constexpr int OutputSFVectorSize = 16;

//TODO: tune
using MmaTileShape_MNK = Shape<_256,_128,_128>;
using ClusterShape_MNK = Shape<_4,_1,_1>;
using PerSmTileShape_MNK = Shape<_256,_128,_128>;

using FusionOperation =
cutlass::epilogue::fusion::qutlass::QutlassLinCombBlockScaleFactorNv<
    OutputSFVectorSize,
    ElementD,
    ElementCompute,
    ElementSFD, LayoutSFDTag,
    ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilderQutlass<
    ArchTag, OperatorClass,
    PerSmTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAutoQutlass,
    //cute::Shape<_32,_64>,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAutoQutlass,
    FusionOperation
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

using FusionOp = typename Gemm::EpilogueOutputOp;
constexpr bool IsBlockScaleSupported = FusionOp::IsBlockScaleSupported;
using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
using LayoutSFD = typename SfdOutputCfg::LayoutSF;

typename Gemm::Arguments args_from_options_nv(torch::Tensor& D,
                                           torch::Tensor& D_sf,
                                           torch::Tensor const& A,
                                           torch::Tensor const& B,
                                           torch::Tensor const& global_scale,
                                           int32_t M, int32_t N, int32_t K)
{
    using ElementA       = typename Gemm::ElementA;
    using ElementB       = typename Gemm::ElementB;
    using ElementC       = typename Gemm::ElementC;
    using ElementD       = typename Gemm::ElementD;
    using ElementCompute = float;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            static_cast<ElementA const*>(A.data_ptr()),      stride_A,
            static_cast<ElementB const*>(B.data_ptr()),      stride_B},
        {
            {1.f, 0.f},
            nullptr, stride_C,
            static_cast<ElementD*>(D.data_ptr()),       stride_D
        }
    };

    if constexpr (IsBlockScaleSupported) {
        arguments.epilogue.thread.block_scale_factor_ptr = static_cast<cutlass::float_ue4m3_t*>(D_sf.data_ptr());
        arguments.epilogue.thread.norm_constant_ptr      = static_cast<ElementAccumulator const*>(global_scale.data_ptr());;
    }

    return arguments;
}

void runGemmNv(torch::Tensor& D,
             torch::Tensor& D_sf,
             torch::Tensor const& A,
             torch::Tensor const& B,
             torch::Tensor const& global_scale,
             int32_t M, int32_t N, int32_t K,
             torch::Device device)
{
    Gemm gemm;

    auto arguments =
        args_from_options_nv(D, D_sf, A, B, global_scale, M, N, K);

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

    CUTLASS_CHECK(gemm.can_implement(arguments));

    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));

    CUTLASS_CHECK(gemm.run(arguments, workspace.get(), stream));
}

void fusedQuantizeNvAbsMax_host_sm100(torch::Tensor& D,
                                      torch::Tensor& D_sf,
                                      torch::Tensor const& A,
                                      torch::Tensor const& B,
                                      torch::Tensor const& global_scale)
{
#if TARGET_CUDA_ARCH == 100 || TARGET_CUDA_ARCH == 103
    int32_t M = A.numel() / 128;
    int32_t N = B.size(1);
    int32_t K = 128;

    runGemmNv(D, D_sf, A, B, global_scale, M, N, K, A.device());
#else
    TORCH_CHECK(false, "Unsupported CUDA arch");
#endif
}

} // namespace QUTLASS