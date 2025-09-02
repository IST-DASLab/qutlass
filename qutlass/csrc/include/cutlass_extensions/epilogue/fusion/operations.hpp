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

#pragma once

#include <cutlass/numeric_conversion.h>
#include <cutlass/layout/matrix.h>
#include <cute/numeric/numeric_types.hpp>
#include <cute/numeric/integral_constant.hpp> // cute::false_type

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass::epilogue::fusion {
namespace qutlass {

using namespace cutlass;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fusion Operations
// Template args must not be implementation dependent
//
/////////////////////////////////////////////////////////////////////////////////////////////////

struct FusionOperation {
  // metadata types/queries that can be overrided
  using ElementOutput = void;
  using ElementCompute = void;
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_indeterminate;

  using ElementSource = void;
  static constexpr bool IsSourceSupported = false;
  static constexpr bool IsResidualSupported = false; // Source is added after activation

  using ElementScalar = void;
  static constexpr int AlignmentScalar = 0;
  static constexpr bool IsScaleFactorSupported = false;
  static constexpr bool IsPerRowScaleSupported = false;
  static constexpr bool IsPerColScaleSupported = false;

  using ElementBias = void;
  static constexpr int AlignmentBias = 0;
  static constexpr bool IsPerRowBiasSupported = false;
  static constexpr bool IsPerColBiasSupported = false;
  static constexpr bool IsDePerRowBiasSupported = false;

  using ActivationFn = void;
  static constexpr bool IsEltActSupported = false;
  static constexpr bool IsDeEltActSupported = false;

  using ElementAux = void;
  using GmemLayoutTagAux = void;
  static constexpr int AlignmentAux = 0;
  static constexpr bool IsAuxOutSupported = false;
  static constexpr bool IsAuxInSupported = false;

  using ElementAmax = void;
  static constexpr bool IsAbsMaxSupported = false;

  using ElementBlockScaleFactor = void;
  static constexpr int SFVecSize = 0;
  static constexpr bool IsBlockScaleSupported = false;               // Umbrella variable to check BlockScaling support in the epilogues
  using GmemLayoutTagScalefactor = void;
};

// D = alpha * acc
template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledAcc : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementScalar_;
  static constexpr int AlignmentScalar = 1;
  static constexpr auto RoundStyle = RoundStyle_;
};

// D = alpha * acc + beta * C
template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinearCombination
    : ScaledAcc<ElementOutput_, ElementCompute_, ElementScalar_, RoundStyle_> {
  using ElementSource = ElementSource_;
  static constexpr bool IsSourceSupported = true;
};

template<
  int SFVecSize_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBlockScaleFactor_,
  class GmemLayoutTagScalefactor_ = cutlass::layout::RowMajor,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct QutlassLinCombBlockScaleFactor
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr bool IsBlockScaleSupported = true;
  using GmemLayoutTagScalefactor = GmemLayoutTagScalefactor_;
};

template<
  int SFVecSize_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBlockScaleFactor_,
  class GmemLayoutTagScalefactor_ = cutlass::layout::RowMajor,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct QutlassLinCombBlockScaleFactorNv
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr bool IsBlockScaleSupported = true;
  using GmemLayoutTagScalefactor = GmemLayoutTagScalefactor_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion
}
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
