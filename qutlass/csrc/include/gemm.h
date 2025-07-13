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
#include <torch/types.h>

void matmul_host_mxf4_bf16_tn(torch::Tensor& D,
                              torch::Tensor const& A,
                              torch::Tensor const& B,
                              torch::Tensor const& A_sf,
                              torch::Tensor const& B_sf,
                              float alpha);

void matmul_host_ada_mxf4_bf16_tn(torch::Tensor const& input,
                                  torch::Tensor const& weight,
                                  torch::Tensor const& input_sf,
                                  torch::Tensor const& weight_sf,
                                  torch::Tensor &out,
                                  float alpha);