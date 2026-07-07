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

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <utility>

#include "include/bindings_utils.h"
#include "include/gemm.h"
#include "include/fused_quantize_host.h"

namespace QUTLASS {

using torch::stable::Tensor;
using torch::headeronly::ScalarType;

Tensor matmul_mxf4_bf16_tn(Tensor const& A,
                          Tensor const& B,
                          Tensor const& A_sf,
                          Tensor const& B_sf,
                          Tensor const& alpha)
{
    check_all_contiguous("matmul_mxf4_bf16_tn", {{A, "A", 0},
                                                 {B, "B", 1},
                                                 {A_sf, "A_sf", 2},
                                                 {B_sf, "B_sf", 3}});
    check_device_type_cuda("matmul_mxf4_bf16_tn", {A, B, A_sf, B_sf, alpha});
    check_all_same_gpu("matmul_mxf4_bf16_tn", {{A, "A", 0},
                                              {B, "B", 1},
                                              {A_sf, "A_sf", 2},
                                              {B_sf, "B_sf", 3},
                                              {alpha, "alpha", 4}});
    STD_TORCH_CHECK(A.scalar_type() == ScalarType::Byte, "A must be uint8");
    STD_TORCH_CHECK(B.scalar_type() == ScalarType::Byte, "B must be uint8");
    STD_TORCH_CHECK(A_sf.scalar_type() == ScalarType::Float8_e8m0fnu,
                    "A_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(B_sf.scalar_type() == ScalarType::Float8_e8m0fnu,
                    "B_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    STD_TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match for A @ B.T");
    STD_TORCH_CHECK(A.size(1) >= 32, "A K-dim must be >= 32");
    STD_TORCH_CHECK(B.size(1) >= 32, "B K-dim must be >= 32");

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto OUT = torch::stable::new_empty(A, {M, N}, ScalarType::BFloat16);

    matmul_host_mxf4_bf16_tn(OUT, A, B, A_sf, B_sf, alpha);

    return OUT;
}

Tensor matmul_nvf4_bf16_tn(Tensor const& A,
                           Tensor const& B,
                           Tensor const& A_sf,
                           Tensor const& B_sf,
                           Tensor const& alpha)
{
    check_all_contiguous("matmul_nvf4_bf16_tn", {{A, "A", 0},
                                                 {B, "B", 1},
                                                 {A_sf, "A_sf", 2},
                                                 {B_sf, "B_sf", 3}});
    check_device_type_cuda("matmul_nvf4_bf16_tn", {A, B, A_sf, B_sf, alpha});
    check_all_same_gpu("matmul_nvf4_bf16_tn", {{A, "A", 0},
                                              {B, "B", 1},
                                              {A_sf, "A_sf", 2},
                                              {B_sf, "B_sf", 3},
                                              {alpha, "alpha", 4}});
    STD_TORCH_CHECK(A.scalar_type() == ScalarType::Byte, "A must be uint8");
    STD_TORCH_CHECK(B.scalar_type() == ScalarType::Byte, "B must be uint8");
    STD_TORCH_CHECK(A_sf.scalar_type() == ScalarType::Float8_e4m3fn,
                    "A_sf must be float8_e4m3fn");
    STD_TORCH_CHECK(B_sf.scalar_type() == ScalarType::Float8_e4m3fn,
                    "B_sf must be float8_e4m3fn");
    STD_TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    STD_TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match for A @ B.T");
    STD_TORCH_CHECK(A.size(1) >= 16, "A K-dim must be >= 16");
    STD_TORCH_CHECK(B.size(1) >= 16, "B K-dim must be >= 16");

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto OUT = torch::stable::new_empty(A, {M, N}, ScalarType::BFloat16);

    matmul_host_nvf4_bf16_tn(OUT, A, B, A_sf, B_sf, alpha);

    return OUT;
}

Tensor matmul_ada_mxf4_bf16_tn(Tensor const& A,
                               Tensor const& B,
                               Tensor const& A_sf,
                               Tensor const& B_sf,
                               Tensor const& alpha)
{
    check_all_contiguous("matmul_ada_mxf4_bf16_tn", {{A, "A", 0},
                                                     {B, "B", 1},
                                                     {A_sf, "A_sf", 2},
                                                     {B_sf, "B_sf", 3}});
    check_device_type_cuda("matmul_ada_mxf4_bf16_tn", {A, B, A_sf, B_sf, alpha});
    check_all_same_gpu("matmul_ada_mxf4_bf16_tn", {{A, "A", 0},
                                                   {B, "B", 1},
                                                   {A_sf, "A_sf", 2},
                                                   {B_sf, "B_sf", 3},
                                                   {alpha, "alpha", 4}});
    STD_TORCH_CHECK(A.scalar_type() == ScalarType::Byte, "A must be uint8");
    STD_TORCH_CHECK(B.scalar_type() == ScalarType::Byte, "B must be uint8");
    STD_TORCH_CHECK(A_sf.scalar_type() == ScalarType::Float8_e8m0fnu,
                    "A_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(B_sf.scalar_type() == ScalarType::Float8_e8m0fnu,
                    "B_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    STD_TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match for A @ B.T");
    STD_TORCH_CHECK(A.size(1) >= 32, "A K-dim must be >= 32");
    STD_TORCH_CHECK(B.size(1) >= 32, "B K-dim must be >= 32");

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto C = torch::stable::new_empty(A, {M, N}, ScalarType::BFloat16);

    matmul_host_ada_mxf4_bf16_tn(A, B, A_sf, B_sf, C, alpha);

    return C;
}

std::tuple<Tensor, Tensor> fusedQuantizeMxQuest(Tensor const& A,
                                                Tensor const& B,
                                                Tensor& OUT,
                                                Tensor& OUT_sf)
{
    check_all_contiguous("fusedQuantizeMxQuest", {{A, "A", 0},
                                                  {B, "B", 1},
                                                  {OUT, "OUT", 2},
                                                  {OUT_sf, "OUT_sf", 3}});
    check_device_type_cuda("fusedQuantizeMxQuest", {A, B, OUT, OUT_sf});
    check_all_same_gpu("fusedQuantizeMxQuest", {{A, "A", 0},
                                              {B, "B", 1},
                                              {OUT, "OUT", 2},
                                              {OUT_sf, "OUT_sf", 3}});
    STD_TORCH_CHECK(A.scalar_type() == ScalarType::BFloat16, "A must be bf16");
    STD_TORCH_CHECK(B.scalar_type() == ScalarType::BFloat16, "B must be bf16");
    STD_TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    STD_TORCH_CHECK((A.numel() % HAD_GS) == 0, "A must be divisible by", HAD_GS);

    if (HAD_GS == 32) {
        fusedQuantizeMxQuest_host(OUT, OUT_sf, A, B);
    } else if (HAD_GS == 64) {
        fusedQuantizeMxQuestHad64_host(OUT, OUT_sf, A, B);
    } else if (HAD_GS == 128) {
        fusedQuantizeMxQuestHad128_host(OUT, OUT_sf, A, B);
    } else {
        STD_TORCH_CHECK(false,
                        "Unsupported rotation size ", HAD_GS,
                        "; expected 32, 64, or 128.");
    }

    return std::make_tuple(OUT, OUT_sf);
}

std::tuple<Tensor, Tensor, Tensor> fusedQuantizeMxQuestWithMask(
    Tensor const& A,
    Tensor const& B,
    Tensor& OUT,
    Tensor& OUT_sf,
    Tensor& OUT_mask)
{
    check_all_contiguous("fusedQuantizeMxQuestWithMask", {{A, "A", 0},
                                                          {B, "B", 1},
                                                          {OUT, "OUT", 2},
                                                          {OUT_sf, "OUT_sf", 3},
                                                          {OUT_mask, "OUT_mask", 4}});
    check_device_type_cuda("fusedQuantizeMxQuestWithMask", {A, B, OUT, OUT_sf, OUT_mask});
    check_all_same_gpu("fusedQuantizeMxQuestWithMask", {{A, "A", 0},
                                                        {B, "B", 1},
                                                        {OUT, "OUT", 2},
                                                        {OUT_sf, "OUT_sf", 3},
                                                        {OUT_mask, "OUT_mask", 4}});
    STD_TORCH_CHECK(A.scalar_type() == ScalarType::BFloat16, "A must be bf16");
    STD_TORCH_CHECK(B.scalar_type() == ScalarType::BFloat16, "B must be bf16");
    STD_TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    STD_TORCH_CHECK((A.numel() % HAD_GS) == 0, "A must be divisible by", HAD_GS);

    if (HAD_GS == 32) {
        fusedQuantizeMxQuestWithMask_host(OUT, OUT_sf, OUT_mask, A, B);
    } else {
        STD_TORCH_CHECK(false,
                        "Unsupported rotation size ", HAD_GS,
                        "; expected 32.");
    }

    return std::make_tuple(OUT, OUT_sf, OUT_mask);
}

std::tuple<Tensor, Tensor> fusedQuantizeMxAbsMax(Tensor const& A,
                                                 Tensor const& B,
                                                 Tensor& OUT,
                                                 Tensor& OUT_sf)
{
    check_all_contiguous("fusedQuantizeMxAbsMax", {{A, "A", 0},
                                                   {B, "B", 1},
                                                   {OUT, "OUT", 2},
                                                   {OUT_sf, "OUT_sf", 3}});
    check_device_type_cuda("fusedQuantizeMxAbsMax", {A, B, OUT, OUT_sf});
    check_all_same_gpu("fusedQuantizeMxAbsMax", {{A, "A", 0},
                                                 {B, "B", 1},
                                                 {OUT, "OUT", 2},
                                                 {OUT_sf, "OUT_sf", 3}});
    STD_TORCH_CHECK(A.scalar_type() == ScalarType::BFloat16, "A must be bf16");
    STD_TORCH_CHECK(B.scalar_type() == ScalarType::BFloat16, "B must be bf16");
    STD_TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    STD_TORCH_CHECK((A.numel() % HAD_GS) == 0, "A must be divisible by", HAD_GS);

    if (HAD_GS == 32) {
        fusedQuantizeMxAbsMax_host(OUT, OUT_sf, A, B);
    } else if (HAD_GS == 64) {
        fusedQuantizeMxAbsMaxHad64_host(OUT, OUT_sf, A, B);
    } else if (HAD_GS == 128) {
#if TARGET_CUDA_ARCH == 100 || TARGET_CUDA_ARCH == 101 || TARGET_CUDA_ARCH == 110
        // FIXME: add input global_scale to interface for consistency
        auto global_scale =
          torch::stable::new_zeros(A, {1}, ScalarType::Float);
        fusedQuantizeMxAbsMax_host_sm100(OUT, OUT_sf, A, B, global_scale);
#elif TARGET_CUDA_ARCH == 120
        fusedQuantizeMxAbsMaxHad128_host(OUT, OUT_sf, A, B);
#endif
    } else {
        STD_TORCH_CHECK(false,
                        "Unsupported rotation size ", HAD_GS,
                        "; expected 32, 64, or 128.");
    }

    return std::make_tuple(OUT, OUT_sf);
}

std::tuple<Tensor, Tensor> fusedQuantizeNvQuest(Tensor const& A,
                                                Tensor const& B,
                                                Tensor& OUT,
                                                Tensor& OUT_sf,
                                                Tensor const& global_scale)
{
    check_all_contiguous("fusedQuantizeNvQuest", {{A, "A", 0},
                                                  {B, "B", 1},
                                                  {OUT, "OUT", 2},
                                                  {OUT_sf, "OUT_sf", 3}});
    check_device_type_cuda("fusedQuantizeNvQuest",
                           {A, B, OUT, OUT_sf, global_scale});
    check_all_same_gpu("fusedQuantizeNvQuest", {{A, "A", 0},
                                                {B, "B", 1},
                                                {OUT, "OUT", 2},
                                                {OUT_sf, "OUT_sf", 3},
                                                {global_scale, "global_scale", 4}});
    STD_TORCH_CHECK(A.scalar_type() == ScalarType::BFloat16, "A must be bf16");
    STD_TORCH_CHECK(B.scalar_type() == ScalarType::BFloat16, "B must be bf16");
    STD_TORCH_CHECK(global_scale.scalar_type() == ScalarType::Float,
                    "global_scale must be float");
    STD_TORCH_CHECK(global_scale.dim() == 1 && global_scale.size(0) == 1,
                    "global_scale must be a scalar");
    STD_TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    STD_TORCH_CHECK((A.numel() % HAD_GS) == 0, "A must be divisible by", HAD_GS);

    if (HAD_GS == 16) {
        fusedQuantizeNvQuest_host(OUT, OUT_sf, A, B, global_scale);
    } else if (HAD_GS == 32) {
        fusedQuantizeNvQuestHad32_host(OUT, OUT_sf, A, B, global_scale);
    } else if (HAD_GS == 64) {
        fusedQuantizeNvQuestHad64_host(OUT, OUT_sf, A, B, global_scale);
    } else if (HAD_GS == 128) {
        fusedQuantizeNvQuestHad128_host(OUT, OUT_sf, A, B, global_scale);
    } else {
        STD_TORCH_CHECK(false,
                        "Unsupported rotation size ", HAD_GS,
                        "; expected 16, 32, 64, or 128.");
    }

    return std::make_tuple(OUT, OUT_sf);
}

std::tuple<Tensor, Tensor> fusedQuantizeNvAbsMax(Tensor const& A,
                                                 Tensor const& B,
                                                 Tensor& OUT,
                                                 Tensor& OUT_sf,
                                                 Tensor const& global_scale)
{
    check_all_contiguous("fusedQuantizeNvAbsMax", {{A, "A", 0},
                                                   {B, "B", 1},
                                                   {OUT, "OUT", 2},
                                                   {OUT_sf, "OUT_sf", 3}});
    check_device_type_cuda("fusedQuantizeNvAbsMax", {A, B, OUT, OUT_sf, global_scale});
    check_all_same_gpu("fusedQuantizeNvAbsMax", {{A, "A", 0},
                                                 {B, "B", 1},
                                                 {OUT, "OUT", 2},
                                                 {OUT_sf, "OUT_sf", 3},
                                                 {global_scale, "global_scale", 4}});
    STD_TORCH_CHECK(A.scalar_type() == ScalarType::BFloat16, "A must be bf16");
    STD_TORCH_CHECK(B.scalar_type() == ScalarType::BFloat16, "B must be bf16");
    STD_TORCH_CHECK(global_scale.scalar_type() == ScalarType::Float,
                    "global_scale must be float");
    STD_TORCH_CHECK(global_scale.dim() == 1 && global_scale.size(0) == 1,
                    "global_scale must be a scalar");
    STD_TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    STD_TORCH_CHECK((A.numel() % HAD_GS) == 0, "A must be divisible by", HAD_GS);

    if (HAD_GS == 16) {
        fusedQuantizeNvAbsMax_host(OUT, OUT_sf, A, B, global_scale);
    } else if (HAD_GS == 32) {
        fusedQuantizeNvAbsMaxHad32_host(OUT, OUT_sf, A, B, global_scale);
    } else if (HAD_GS == 64) {
        fusedQuantizeNvAbsMaxHad64_host(OUT, OUT_sf, A, B, global_scale);
    } else if(HAD_GS==128){
#if TARGET_CUDA_ARCH == 100 || TARGET_CUDA_ARCH == 101 || TARGET_CUDA_ARCH == 110
        fusedQuantizeNvAbsMax_host_sm100(OUT, OUT_sf, A, B, global_scale);
#elif TARGET_CUDA_ARCH == 120
        fusedQuantizeNvAbsMaxHad128_host(OUT, OUT_sf, A, B, global_scale);
#endif
    } else {
        STD_TORCH_CHECK(false,
                        "Unsupported rotation size ", HAD_GS,
                        "; expected 16, 32, 64, or 128.");
    }

    return std::make_tuple(OUT, OUT_sf);
}

}  // namespace QUTLASS

STABLE_TORCH_LIBRARY_FRAGMENT(_qutlass_C, ops) {
  ops.def("matmul_mxf4_bf16_tn(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
  ops.def("matmul_nvf4_bf16_tn(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
  ops.def("matmul_ada_mxf4_bf16_tn(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
  ops.def("fusedQuantizeMxQuest(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf) -> (Tensor, Tensor)");
  ops.def("fusedQuantizeMxQuestWithMask(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf, Tensor OUT_mask) -> (Tensor, Tensor, Tensor)");
  ops.def("fusedQuantizeMxAbsMax(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf) -> (Tensor, Tensor)");
  ops.def("fusedQuantizeNvQuest(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf, Tensor global_scale) -> (Tensor, Tensor)");
  ops.def("fusedQuantizeNvAbsMax(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf, Tensor global_scale) -> (Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(_qutlass_C, CUDA, ops) {
  ops.impl("matmul_mxf4_bf16_tn", TORCH_BOX(&QUTLASS::matmul_mxf4_bf16_tn));
  ops.impl("matmul_nvf4_bf16_tn", TORCH_BOX(&QUTLASS::matmul_nvf4_bf16_tn));
  ops.impl("matmul_ada_mxf4_bf16_tn", TORCH_BOX(&QUTLASS::matmul_ada_mxf4_bf16_tn));
  ops.impl("fusedQuantizeMxQuest", TORCH_BOX(&QUTLASS::fusedQuantizeMxQuest));
  ops.impl("fusedQuantizeMxQuestWithMask", TORCH_BOX(&QUTLASS::fusedQuantizeMxQuestWithMask));
  ops.impl("fusedQuantizeMxAbsMax", TORCH_BOX(&QUTLASS::fusedQuantizeMxAbsMax));
  ops.impl("fusedQuantizeNvQuest", TORCH_BOX(&QUTLASS::fusedQuantizeNvQuest));
  ops.impl("fusedQuantizeNvAbsMax", TORCH_BOX(&QUTLASS::fusedQuantizeNvAbsMax));
}
