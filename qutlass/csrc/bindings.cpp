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

#include <vector>
#include <iostream>
#include <utility>

#include "include/gemm.h"
#include "include/fused_quantize_host.h"


namespace QUTLASS {

torch::Tensor matmul_mxf4_bf16_tn(torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf,
                                  torch::Tensor const& alpha)
{
    torch::checkAllContiguous("matmul_mxf4_bf16_tn", {{A, "A", 0},
                                                      {B, "B", 1},
                                                      {A_sf, "A_sf", 2},
                                                      {B_sf, "B_sf", 3}});
    torch::checkDeviceType("matmul_mxf4_bf16_tn", {A, B, A_sf, B_sf, alpha}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("matmul_mxf4_bf16_tn", {{A, "A", 0},
                                                   {B, "B", 1},
                                                   {A_sf, "A_sf", 2},
                                                   {B_sf, "B_sf", 3},
                                                   {alpha, "alpha", 4}});
    TORCH_CHECK(A.scalar_type() == at::kByte, "A must be uint8");
    TORCH_CHECK(B.scalar_type() == at::kByte, "B must be uint8");
    TORCH_CHECK(A_sf.scalar_type() == at::kFloat8_e8m0fnu, "A_sf must be float8_e8m0fnu");
    TORCH_CHECK(B_sf.scalar_type() == at::kFloat8_e8m0fnu, "B_sf must be float8_e8m0fnu");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match for A @ B.T");
    TORCH_CHECK(A.size(1) >= 32, "A K-dim must be >= 32");
    TORCH_CHECK(B.size(1) >= 32, "B K-dim must be >= 32");

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto OUT   = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

    matmul_host_mxf4_bf16_tn(OUT, A, B, A_sf, B_sf, alpha);

    return OUT;
}

torch::Tensor matmul_nvf4_bf16_tn(torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf,
                                  torch::Tensor const& alpha)
{
    torch::checkAllContiguous("matmul_nvf4_bf16_tn", {{A, "A", 0},
                                                      {B, "B", 1},
                                                      {A_sf, "A_sf", 2},
                                                      {B_sf, "B_sf", 3}});
    torch::checkDeviceType("matmul_nvf4_bf16_tn", {A, B, A_sf, B_sf, alpha}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("matmul_nvf4_bf16_tn", {{A, "A", 0},
                                                   {B, "B", 1},
                                                   {A_sf, "A_sf", 2},
                                                   {B_sf, "B_sf", 3},
                                                   {alpha, "alpha", 4}});
    TORCH_CHECK(A.scalar_type() == at::kByte, "A must be uint8");
    TORCH_CHECK(B.scalar_type() == at::kByte, "B must be uint8");
    TORCH_CHECK(A_sf.scalar_type() == at::kFloat8_e4m3fn, "A_sf must be float8_e4m3fn");
    TORCH_CHECK(B_sf.scalar_type() == at::kFloat8_e4m3fn, "B_sf must be float8_e4m3fn");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match for A @ B.T");
    TORCH_CHECK(A.size(1) >= 16, "A K-dim must be >= 16");
    TORCH_CHECK(B.size(1) >= 16, "B K-dim must be >= 16");

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto OUT   = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

    matmul_host_nvf4_bf16_tn(OUT, A, B, A_sf, B_sf, alpha);

    return OUT;
}

torch::Tensor matmul_ada_mxf4_bf16_tn(torch::Tensor const&A,
                                      torch::Tensor const&B,
                                      torch::Tensor const&A_sf,
                                      torch::Tensor const&B_sf,
                                      torch::Tensor const& alpha)
{
    torch::checkAllContiguous("matmul_ada_mxf4_bf16_tn", {{A, "A", 0},
                                                      {B, "B", 1},
                                                      {A_sf, "A_sf", 2},
                                                      {B_sf, "B_sf", 3}});
    torch::checkDeviceType("matmul_ada_mxf4_bf16_tn", {A, B, A_sf, B_sf, alpha}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("matmul_ada_mxf4_bf16_tn", {{A, "A", 0},
                                                       {B, "B", 1},
                                                       {A_sf, "A_sf", 2},
                                                       {B_sf, "B_sf", 3},
                                                       {alpha, "alpha", 4}});
    TORCH_CHECK(A.scalar_type() == at::kByte, "A must be uint8");
    TORCH_CHECK(B.scalar_type() == at::kByte, "B must be uint8");
    TORCH_CHECK(A_sf.scalar_type() == at::kFloat8_e8m0fnu, "A_sf must be float8_e8m0fnu");
    TORCH_CHECK(B_sf.scalar_type() == at::kFloat8_e8m0fnu, "B_sf must be float8_e8m0fnu");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match for A @ B.T");
    TORCH_CHECK(A.size(1) >= 32, "A K-dim must be >= 32");
    TORCH_CHECK(B.size(1) >= 32, "B K-dim must be >= 32");

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

    matmul_host_ada_mxf4_bf16_tn(A, B, A_sf, B_sf, C, alpha);

    return C;
}

std::tuple<torch::Tensor, torch::Tensor> fusedQuantizeMxQuest(torch::Tensor const& A,
                                                              torch::Tensor const& B,
                                                              torch::Tensor& OUT,
                                                              torch::Tensor& OUT_sf)
{
    torch::checkAllContiguous("fusedQuantizeMxQuest", {{A, "A", 0},
                                                       {B, "B", 1},
                                                       {OUT, "OUT", 2},
                                                       {OUT_sf, "OUT_sf", 3}});
    torch::checkDeviceType("fusedQuantizeMxQuest", {A, B, OUT, OUT_sf}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("fusedQuantizeMxQuest", {{A, "A", 0},
                                                    {B, "B", 1},
                                                    {OUT, "OUT", 2},
                                                    {OUT_sf, "OUT_sf", 3}});
    TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.scalar_type() == at::kBFloat16, "B must be bf16");
    TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    TORCH_CHECK((A.numel()%HAD_GS)==0, "A must be divisible by", HAD_GS);

    if(HAD_GS==32){
        fusedQuantizeMxQuest_host(OUT, OUT_sf, A, B);
    } else if(HAD_GS==64){
        fusedQuantizeMxQuestHad64_host(OUT, OUT_sf, A, B);
    } else if(HAD_GS==128){
        fusedQuantizeMxQuestHad128_host(OUT, OUT_sf, A, B);
    } else {
        TORCH_CHECK(false,
                    "Unsupported rotation size ", HAD_GS,
                    "; expected 32, 64, or 128.");
    }

    return std::make_tuple(OUT, OUT_sf);
}

std::tuple<torch::Tensor, torch::Tensor> fusedQuantizeMxAbsMax(torch::Tensor const& A,
                                                               torch::Tensor const& B,
                                                               torch::Tensor& OUT,
                                                               torch::Tensor& OUT_sf)
{
    torch::checkAllContiguous("fusedQuantizeMxAbsMax", {{A, "A", 0},
                                                        {B, "B", 1},
                                                        {OUT, "OUT", 2},
                                                        {OUT_sf, "OUT_sf", 3}});
    torch::checkDeviceType("fusedQuantizeMxAbsMax", {A, B, OUT, OUT_sf}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("fusedQuantizeMxAbsMax", {{A, "A", 0},
                                                     {B, "B", 1},
                                                     {OUT, "OUT", 2},
                                                     {OUT_sf, "OUT_sf", 3}});
    TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.scalar_type() == at::kBFloat16, "B must be bf16");
    TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    TORCH_CHECK((A.numel()%HAD_GS)==0, "A must be divisible by", HAD_GS);

    if(HAD_GS==32){
        fusedQuantizeMxAbsMax_host(OUT, OUT_sf, A, B);
    } else if(HAD_GS==64){
        fusedQuantizeMxAbsMaxHad64_host(OUT, OUT_sf, A, B);
    } else if(HAD_GS==128){
#if TARGET_CUDA_ARCH == 100
        auto opts = torch::TensorOptions().dtype(torch::kFloat).device(A.device());
        auto global_scale = torch::tensor(0.0f, opts); //FIXME: add input global_scale to interface for consistency
        fusedQuantizeMxAbsMax_host_sm100(OUT, OUT_sf, A, B, global_scale);
#elif TARGET_CUDA_ARCH == 120
        fusedQuantizeMxAbsMaxHad128_host(OUT, OUT_sf, A, B);
#endif

    } else {
        TORCH_CHECK(false,
                    "Unsupported rotation size ", HAD_GS,
                    "; expected 32, 64, or 128.");
    }

    return std::make_tuple(OUT, OUT_sf);
}

std::tuple<torch::Tensor, torch::Tensor> fusedQuantizeNv(torch::Tensor const& A,
                                                         torch::Tensor const& B,
                                                         torch::Tensor& OUT,
                                                         torch::Tensor& OUT_sf,
                                                         torch::Tensor const& global_scale)
{
    torch::checkAllContiguous("fusedQuantizeNv", {{A, "A", 0},
                                                  {B, "B", 1},
                                                  {OUT, "OUT", 2},
                                                  {OUT_sf, "OUT_sf", 3}});
    torch::checkDeviceType("fusedQuantizeNv", {A, B, OUT, OUT_sf, global_scale}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("fusedQuantizeNv", {{A, "A", 0},
                                               {B, "B", 1},
                                               {OUT, "OUT", 2},
                                               {OUT_sf, "OUT_sf", 3},
                                               {global_scale, "global_scale", 4}});
    TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.scalar_type() == at::kBFloat16, "B must be bf16");
    TORCH_CHECK(global_scale.scalar_type() == at::kFloat, "global_scale must be float");
    TORCH_CHECK(global_scale.dim() == 1 && global_scale.size(0) == 1, "global_scale must be a scalar");
    TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    TORCH_CHECK((A.numel()%HAD_GS)==0, "A must be divisible by", HAD_GS);

    if(HAD_GS==16){
        fusedQuantizeNv_host(OUT, OUT_sf, A, B, global_scale);
    } else if(HAD_GS==32){
        fusedQuantizeNvHad32_host(OUT, OUT_sf, A, B, global_scale);
    } else if(HAD_GS==64){
        fusedQuantizeNvHad64_host(OUT, OUT_sf, A, B, global_scale);
    } else if(HAD_GS==128){
#if TARGET_CUDA_ARCH == 100
        fusedQuantizeNvAbsMax_host_sm100(OUT, OUT_sf, A, B, global_scale);
#elif TARGET_CUDA_ARCH == 120
        fusedQuantizeNvHad128_host(OUT, OUT_sf, A, B, global_scale);
#endif
    } else {
        TORCH_CHECK(false,
                    "Unsupported rotation size ", HAD_GS,
                    "; expected 16, 32, 64, or 128.");
    }

    return std::make_tuple(OUT, OUT_sf);
}

TORCH_LIBRARY(_qutlass_C, m) {
  m.def("matmul_mxf4_bf16_tn(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
  m.def("matmul_nvf4_bf16_tn(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
  m.def("matmul_ada_mxf4_bf16_tn(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");

  m.def("fusedQuantizeMxQuest(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf) -> (Tensor, Tensor)");
  m.def("fusedQuantizeMxAbsMax(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf) -> (Tensor, Tensor)");
  m.def("fusedQuantizeNv(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf, Tensor global_scale) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(_qutlass_C, CUDA, m) {
  m.impl("matmul_mxf4_bf16_tn",      TORCH_FN(QUTLASS::matmul_mxf4_bf16_tn));
  m.impl("matmul_nvf4_bf16_tn",      TORCH_FN(QUTLASS::matmul_nvf4_bf16_tn));
  m.impl("matmul_ada_mxf4_bf16_tn",  TORCH_FN(QUTLASS::matmul_ada_mxf4_bf16_tn));

  m.impl("fusedQuantizeMxQuest",     TORCH_FN(QUTLASS::fusedQuantizeMxQuest));
  m.impl("fusedQuantizeMxAbsMax",    TORCH_FN(QUTLASS::fusedQuantizeMxAbsMax));
  m.impl("fusedQuantizeNv",          TORCH_FN(QUTLASS::fusedQuantizeNv));
}

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

#ifndef QUTLASS_DISABLE_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{
    m.def("matmul_mxf4_bf16_tn",     &matmul_mxf4_bf16_tn,     "matmul_mxf4_bf16_tn");
    m.def("matmul_ada_mxf4_bf16_tn", &matmul_ada_mxf4_bf16_tn, "matmul_ada_mxf4_bf16_tn");
    m.def("matmul_nvf4_bf16_tn",     &matmul_nvf4_bf16_tn,     "matmul_nvf4_bf16_tn");
    m.def("fusedQuantizeMxQuest",  &QUTLASS::fusedQuantizeMxQuest,  "fusedQuantizeMxQuest");
    m.def("fusedQuantizeMxAbsMax", &QUTLASS::fusedQuantizeMxAbsMax, "fusedQuantizeMxAbsMax");
    m.def("fusedQuantizeNv",       &QUTLASS::fusedQuantizeNv,       "fusedQuantizeNv");
}
#endif
}