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

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include <gemm.h>

using namespace cute;

// TODO(later): move somewhere else?
using ElementD = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

template <typename PerSmTileShape_MNK, typename ClusterShape, typename ArchTag>
using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        PerSmTileShape_MNK, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
        >::CollectiveOp;

template <typename MmaTileShape, typename ClusterShape, typename PerSmTileShape_MNK,
          typename ArchTag,
          typename ElementA, typename LayoutATag, int AlignmentA,
          typename ElementB, typename LayoutBTag, int AlignmentB>
using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(
            sizeof(typename CollectiveEpilogue<PerSmTileShape_MNK,
                                               ClusterShape,
                                               ArchTag>::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
        >::CollectiveOp;

template <typename MmaTileShape, typename ClusterShape, typename PerSmTileShape_MNK,
          typename ArchTag,
          typename ElementA, typename LayoutATag, int AlignmentA,
          typename ElementB, typename LayoutBTag, int AlignmentB>
using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop<MmaTileShape, ClusterShape, PerSmTileShape_MNK,
                           ArchTag,
                           ElementA, LayoutATag, AlignmentA,
                           ElementB, LayoutBTag, AlignmentB>,
        CollectiveEpilogue<PerSmTileShape_MNK, ClusterShape, ArchTag>,
        void>;

// TODO(later): fix single-template issue
template <typename MmaTileShape, typename ClusterShape, typename PerSmTileShape_MNK,
          typename ArchTag,
          typename ElementA, typename LayoutATag, int AlignmentA,
          typename ElementB, typename LayoutBTag, int AlignmentB>
using Gemm_ =
    cutlass::gemm::device::GemmUniversalAdapter<
                    GemmKernel<MmaTileShape, ClusterShape, PerSmTileShape_MNK,
                               ArchTag,
                               ElementA, LayoutATag, AlignmentA,
                               ElementB, LayoutBTag, AlignmentB>>;

template <typename Gemm, typename ScaleType>
typename Gemm::Arguments args_from_options(
                                at::Tensor& D,
                                at::Tensor const& A,
                                at::Tensor const& B,
                                at::Tensor const& A_sf,
                                at::Tensor const& B_sf,
                                float alpha,
                                int M, int N, int K)
{
    using ElementA       = typename Gemm::ElementA;
    using ElementB       = typename Gemm::ElementB;
    using ElementD       = typename Gemm::ElementD;
    using ElementSFA     = ScaleType;
    using ElementSFB     = ScaleType;
    using ElementCompute = float;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    using Sm1xxBlkScaledConfig =
        typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            static_cast<ElementA const*>(A.data_ptr()),      stride_A,
            static_cast<ElementB const*>(B.data_ptr()),      stride_B,
            static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
            static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
        {
            {alpha, 0.f},
            static_cast<ElementD const*>(D.data_ptr()), stride_D,
            static_cast<ElementD*>(D.data_ptr()),       stride_D
        }
    };

    return arguments;
}

template <typename Gemm, typename ScaleType>
void runGemm(at::Tensor& D,
             at::Tensor const& A,
             at::Tensor const& B,
             at::Tensor const& A_sf,
             at::Tensor const& B_sf,
             float alpha,
             int M, int N, int K,
             torch::Device device)
{
    Gemm gemm;

    auto arguments =
        args_from_options<Gemm, ScaleType>(D, A, B, A_sf, B_sf, alpha, M, N, K);

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

    CUTLASS_CHECK(gemm.can_implement(arguments));

    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));

    CUTLASS_CHECK(gemm.run(arguments, workspace.get(), stream));
}

void matmul_host_mxf4_bf16_tn(torch::Tensor& D,
                              torch::Tensor const& A,
                              torch::Tensor const& B,
                              torch::Tensor const& A_sf,
                              torch::Tensor const& B_sf,
                              float alpha)
{
    auto const m = A.sizes()[0];
    auto const n = B.sizes()[0];
    auto const k = A.sizes()[1] * 2;

    using ElementA   = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 128;

    using ElementB   = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 128;

    using ArchTag = cutlass::arch::Sm120;
    using ClusterShape       = Shape<_1,_1,_1>;
    if(m<512){
        using MmaTileShape       = Shape<_128,_128,_128>;
        using PerSmTileShape_MNK = Shape<_128,_128,_128>;

        runGemm<Gemm_<MmaTileShape, ClusterShape, PerSmTileShape_MNK,
                    ArchTag,
                    ElementA, LayoutATag, AlignmentA,
                    ElementB, LayoutBTag, AlignmentB>, cutlass::float_ue8m0_t
                >(D, A, B, A_sf, B_sf, alpha, m, n, k, A.device());
    } else {
        using MmaTileShape       = Shape<_256,_128,_128>;
        using PerSmTileShape_MNK = Shape<_256,_128,_128>;

        runGemm<Gemm_<MmaTileShape, ClusterShape, PerSmTileShape_MNK,
                    ArchTag,
                    ElementA, LayoutATag, AlignmentA,
                    ElementB, LayoutBTag, AlignmentB>, cutlass::float_ue8m0_t
                >(D, A, B, A_sf, B_sf, alpha, m, n, k, A.device());
    }
}