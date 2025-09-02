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
  \brief Fusion callbacks specializations for the sm90 TMA warp-specialized (ws) epilogue
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass_extensions/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"

#include "cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {
namespace qutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class NodeOp, class... ChildOps>
using Sm90EVT = Sm90TreeVisitor<NodeOp, ChildOps...>;

// D = alpha * acc
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    cutlass::epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    cutlass::epilogue::fusion::qutlass::ScaledAcc<ElementOutput, ElementCompute, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90EVT<Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute, RoundStyle>,
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>,
      Sm90AccFetch
    > {
  using Impl =
    Sm90EVT<Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute, RoundStyle>,
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>,
      Sm90AccFetch
    >;
  using Operation = cutlass::epilogue::fusion::qutlass::ScaledAcc<ElementOutput, ElementCompute, ElementScalar, RoundStyle>;

  struct Arguments {
    // Give a name and flat ordering to the fusion callback args
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};

    // Conversion to the args expected by the visitor implementation
    // to_underlying_arguments will implicitly call this
    operator typename Impl::Arguments() const {
      return
        {    // binary op : alpha * acc
          {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
          {},                     // leaf args : acc
          {} // binary args : multiplies
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C
template<
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinearCombination =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
    Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // beta
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
      Sm90AccFetch // acc
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    cutlass::epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    cutlass::epilogue::fusion::qutlass::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Operation = cutlass::epilogue::fusion::qutlass::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {    // ternary op : beta * C + (alpha * acc)
          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // binary op : alpha * acc
            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {}                  // binary args : multiplies
          },                    // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};



/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {
template <class FusionOpOrCallbacks, class = cute::void_t<>>
struct get_element_aux {
  using type = void;
};

template <class FusionOpOrCallbacks>
struct get_element_aux<FusionOpOrCallbacks, cute::void_t<typename FusionOpOrCallbacks::ElementAux>> {
  using type = typename FusionOpOrCallbacks::ElementAux;
};

template <class NodeOp, class... ChildOps>
struct get_element_aux<Sm90TreeVisitor<NodeOp, ChildOps...>, cute::void_t<>> {
  using type = typename get_element_aux<NodeOp>::type;
};

template <class... Ts>
struct get_element_aux<FusionCallbacks<Ts...>, cute::void_t<typename FusionCallbacks<Ts...>::Operation>> {
 private:
  using Operation = typename FusionCallbacks<Ts...>::Operation;
 public:
  using type = typename get_element_aux<Operation>::type;
};
} // namespace cutlass:epilogue::fusion::detail

template <class Callbacks>
using get_element_aux_t = typename detail::get_element_aux<Callbacks>::type;

} // namespace cutlass::epilogue::fusion
}
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
