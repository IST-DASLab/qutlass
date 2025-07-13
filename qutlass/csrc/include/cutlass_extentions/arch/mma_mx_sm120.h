/*
 * Modified by Roberto L. Castro (Roberto.LopezCastro@ist.ac.at).
*/

/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Matrix multiply
*/

#pragma once

#include <cuda/std/cassert>

#include "cutlass/cutlass.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

////////////////////////////////////////////////////////////////////////////////

#if ((__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))

#define CUTLASS_ARCH_MMA_SM120_SUPPORTED 1

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200))
#define CUTLASS_ARCH_MMA_SM120_ENABLED

#endif

#endif

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::float_e2m1_t,
  layout::RowMajor,
  cutlass::float_e2m1_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::float_e2m1_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::float_e2m1_t, 32>;

  using ElementB = cutlass::float_e2m1_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::float_e2m1_t, 16>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    cutlass::float_ue8m0_t* ascale,
    cutlass::float_ue8m0_t* bscale
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM120_ENABLED)

    uint32_t const * A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const * B = reinterpret_cast<uint32_t const *>(&b);

    float const *C = reinterpret_cast<float const *>(&c);
    float *D       = reinterpret_cast<float *>(&d);

    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    uint32_t sfa0 = *(uint32_t*)ascale;
    uint32_t sfb0 = *(uint32_t*)bscale;

    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      :  "=f"(D[0]),  "=f"(D[1]),  "=f"(D[2]),  "=f"(D[3])
      :   "r"(A[0]),   "r"(A[1]),   "r"(A[2]),   "r"(A[3]),
          "r"(B[0]),   "r"(B[1]),
          "f"(C[0]),   "f"(C[1]),   "f"(C[2]),   "f"(C[3]),
          "r"(sfa0),  "h"(bidA), "h"(tidA),
          "r"(sfb0),  "h"(bidB), "h"(tidB));

#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

} // namespace arch
} // namespace cutlass
/////////////////////////////////////////////////////////////////////////////////////////////////
