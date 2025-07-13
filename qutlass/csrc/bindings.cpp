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

#include <torch/extension.h>
#include <gemm.h>
#include "fused_quantize_host.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <utility>


namespace QUTLASS {

torch::Tensor matmul_mxf4_bf16_tn(torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf,
                                  float alpha)
{
    torch::checkAllContiguous("matmul_mxf4_bf16_tn", {{A, "A", 0},
                                                      {B, "B", 1},
                                                      {A_sf, "A_sf", 2},
                                                      {B_sf, "B_sf", 3}});
    torch::checkDeviceType("matmul_mxf4_bf16_tn", {A, B, A_sf, B_sf}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("matmul_mxf4_bf16_tn", {{A, "A", 0},
                                                   {B, "B", 1},
                                                   {A_sf, "A_sf", 2},
                                                   {B_sf, "B_sf", 3}});
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
    auto OUT = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

    matmul_host_mxf4_bf16_tn(OUT, A, B, A_sf, B_sf, alpha);

    return OUT;
}

torch::Tensor matmul_ada_mxf4_bf16_tn(torch::Tensor const&A,
                                      torch::Tensor const&B,
                                      torch::Tensor const&A_sf,
                                      torch::Tensor const&B_sf,
                                      float alpha)
{
    torch::checkAllContiguous("matmul_ada_mxf4_bf16_tn", {{A, "A", 0},
                                                      {B, "B", 1},
                                                      {A_sf, "A_sf", 2},
                                                      {B_sf, "B_sf", 3}});
    torch::checkDeviceType("matmul_ada_mxf4_bf16_tn", {A, B, A_sf, B_sf}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("matmul_ada_mxf4_bf16_tn", {{A, "A", 0},
                                                       {B, "B", 1},
                                                       {A_sf, "A_sf", 2},
                                                       {B_sf, "B_sf", 3}});
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
    TORCH_CHECK((A.numel()%32)==0, "A must be divisible by 32");
    TORCH_CHECK(B.size(0)==32 && B.size(1)==32, "Rotation matrix must be of size 32x32");

    fusedQuantizeMxQuest_host(OUT, OUT_sf, A, B);

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
    TORCH_CHECK((A.numel()%32)==0, "A must be divisible by 32");
    TORCH_CHECK((B.numel()%32)==0, "B must be divisible by 32");
    TORCH_CHECK(B.size(0)==32 && B.size(1)==32, "Rotation matrix must be of size 32x32");

    fusedQuantizeMxAbsMax_host(OUT, OUT_sf, A, B);

    return std::make_tuple(OUT, OUT_sf);
}

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{
    m.def("matmul_mxf4_bf16_tn", &matmul_mxf4_bf16_tn,
        "matmul_mxf4_bf16_tn");
    m.def("matmul_ada_mxf4_bf16_tn", &matmul_ada_mxf4_bf16_tn,
        "matmul_ada_mxf4_bf16_tn");
    m.def("fusedQuantizeMxQuest", &QUTLASS::fusedQuantizeMxQuest,
        "fusedQuantizeMxQuest");
    m.def("fusedQuantizeMxAbsMax", &QUTLASS::fusedQuantizeMxAbsMax,
        "fusedQuantizeMxAbsMax");
}
}