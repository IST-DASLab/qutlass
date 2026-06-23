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

#pragma once
#include <common.h>

void matmul_host_mxf4_bf16_tn(torch::stable::Tensor& D,
                              torch::stable::Tensor const& A,
                              torch::stable::Tensor const& B,
                              torch::stable::Tensor const& A_sf,
                              torch::stable::Tensor const& B_sf,
                              torch::stable::Tensor const& alpha);

void matmul_host_ada_mxf4_bf16_tn(torch::stable::Tensor const& input,
                                  torch::stable::Tensor const& weight,
                                  torch::stable::Tensor const& input_sf,
                                  torch::stable::Tensor const& weight_sf,
                                  torch::stable::Tensor &out,
                                  torch::stable::Tensor const& alpha);

void matmul_host_nvf4_bf16_tn(torch::stable::Tensor& D,
                              torch::stable::Tensor const& A,
                              torch::stable::Tensor const& B,
                              torch::stable::Tensor const& A_sf,
                              torch::stable::Tensor const& B_sf,
                              torch::stable::Tensor const& alpha);

void matmul_host_mxf8_bf16_tn(torch::stable::Tensor& D,
                              torch::stable::Tensor const& A,
                              torch::stable::Tensor const& B,
                              torch::stable::Tensor const& A_sf,
                              torch::stable::Tensor const& B_sf,
                              torch::stable::Tensor const& alpha);

void matmul_host_mxf8_bf16_nn(torch::stable::Tensor& D,
                              torch::stable::Tensor const& A,
                              torch::stable::Tensor const& B,
                              torch::stable::Tensor const& A_sf,
                              torch::stable::Tensor const& B_sf,
                              torch::stable::Tensor const& alpha);