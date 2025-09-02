/*
 * Modified by Roberto L. Castro (Roberto.LopezCastro@ist.ac.at).
*/

/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
  \brief Fusion callbacks specializations for the sm100 TMA warp-specialized (ws) epilogue
*/


#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"

#include "cutlass_extensions/epilogue/fusion/callbacks.hpp"
#include "cutlass_extensions/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass_extensions/epilogue/fusion/sm100_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass_extensions/epilogue/fusion/sm100_visitor_store_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass::epilogue::fusion {

namespace qutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////
// D = alpha * acc + beta * C
// With Row BlockScaleFactor Generation.
template<
  int SFVecsize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100QutlassLinearCombRowBlockScaleFactor =
  Sm90EVT<Sm100BlockScaleFactorRowStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>, // gen scalefactor
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    cutlass::epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    cutlass::epilogue::fusion::qutlass::QutlassLinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100QutlassLinearCombRowBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100QutlassLinearCombRowBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = cutlass::epilogue::fusion::qutlass::QutlassLinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {
          {
            // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}                    // ternary args : multiply_add
          },
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  int SFVecsize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100QutlassLinearCombRowBlockScaleFactorNv =
  Sm90EVT<Sm100BlockScaleFactorRowStoreNv<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>, // gen scalefactor
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    cutlass::epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    cutlass::epilogue::fusion::qutlass::QutlassLinCombBlockScaleFactorNv<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100QutlassLinearCombRowBlockScaleFactorNv<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100QutlassLinearCombRowBlockScaleFactorNv<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = cutlass::epilogue::fusion::qutlass::QutlassLinCombBlockScaleFactorNv<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {
          {
            // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}                    // ternary args : multiply_add
          },
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

} // namespace cutlass::epilogue::fusion
}
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
