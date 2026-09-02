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

namespace QUTLASS {

void fusedQuantizeMxQuest_host(torch::stable::Tensor&       D,
                               torch::stable::Tensor&       D_sf,
                               torch::stable::Tensor const& A,
                               torch::stable::Tensor const& B,
                               bool is_sf_swizzled_layout);

void fusedQuantizeMxQuestWithMask_host(torch::stable::Tensor&       D,
                                       torch::stable::Tensor&       D_sf,
                                       torch::stable::Tensor&       D_mask,
                                       torch::stable::Tensor const& A,
                                       torch::stable::Tensor const& B,
                                       bool is_sf_swizzled_layout);

void fusedQuantizeMxAbsMax_host(torch::stable::Tensor&       D,
                                torch::stable::Tensor&       D_sf,
                                torch::stable::Tensor const& A,
                                torch::stable::Tensor const& B,
                                torch::stable::Tensor const& global_scale,
                                bool is_sf_swizzled_layout);

void fusedQuantizeMxQuestHad64_host(torch::stable::Tensor&       D,
                                    torch::stable::Tensor&       D_sf,
                                    torch::stable::Tensor const& A,
                                    torch::stable::Tensor const& B,
                                    bool is_sf_swizzled_layout);

void fusedQuantizeMxAbsMaxHad64_host(torch::stable::Tensor&       D,
                                     torch::stable::Tensor&       D_sf,
                                     torch::stable::Tensor const& A,
                                     torch::stable::Tensor const& B,
                                     torch::stable::Tensor const& global_scale,
                                     bool is_sf_swizzled_layout);

void fusedQuantizeMxQuestHad128_host(torch::stable::Tensor&       D,
                                     torch::stable::Tensor&       D_sf,
                                     torch::stable::Tensor const& A,
                                     torch::stable::Tensor const& B,
                                     bool is_sf_swizzled_layout);

void fusedQuantizeMxAbsMaxHad128_host(torch::stable::Tensor&       D,
                                      torch::stable::Tensor&       D_sf,
                                      torch::stable::Tensor const& A,
                                      torch::stable::Tensor const& B,
                                      torch::stable::Tensor const& global_scale,
                                      bool is_sf_swizzled_layout);

void fusedQuantizeNvQuest_host(torch::stable::Tensor&       D,
                               torch::stable::Tensor&       D_sf,
                               torch::stable::Tensor const& A,
                               torch::stable::Tensor const& B,
                               torch::stable::Tensor const& global_scale,
                               bool is_sf_swizzled_layout);

void fusedQuantizeNvQuestHad32_host(torch::stable::Tensor&       D,
                                    torch::stable::Tensor&       D_sf,
                                    torch::stable::Tensor const& A,
                                    torch::stable::Tensor const& B,
                                    torch::stable::Tensor const& global_scale,
                                    bool is_sf_swizzled_layout);

void fusedQuantizeNvQuestHad64_host(torch::stable::Tensor&       D,
                                    torch::stable::Tensor&       D_sf,
                                    torch::stable::Tensor const& A,
                                    torch::stable::Tensor const& B,
                                    torch::stable::Tensor const& global_scale,
                                    bool is_sf_swizzled_layout);

void fusedQuantizeNvQuestHad128_host(torch::stable::Tensor&       D,
                                     torch::stable::Tensor&       D_sf,
                                     torch::stable::Tensor const& A,
                                     torch::stable::Tensor const& B,
                                     torch::stable::Tensor const& global_scale,
                                     bool is_sf_swizzled_layout);

void fusedQuantizeNvAbsMax_host(torch::stable::Tensor&       D,
                                torch::stable::Tensor&       D_sf,
                                torch::stable::Tensor const& A,
                                torch::stable::Tensor const& B,
                                torch::stable::Tensor const& global_scale,
                                bool is_sf_swizzled_layout);

void fusedQuantizeNvAbsMaxHad32_host(torch::stable::Tensor&       D,
                                     torch::stable::Tensor&       D_sf,
                                     torch::stable::Tensor const& A,
                                     torch::stable::Tensor const& B,
                                     torch::stable::Tensor const& global_scale,
                                     bool is_sf_swizzled_layout);

void fusedQuantizeNvAbsMaxHad64_host(torch::stable::Tensor&       D,
                                     torch::stable::Tensor&       D_sf,
                                     torch::stable::Tensor const& A,
                                     torch::stable::Tensor const& B,
                                     torch::stable::Tensor const& global_scale,
                                     bool is_sf_swizzled_layout);

void fusedQuantizeNvAbsMaxHad128_host(torch::stable::Tensor&       D,
                                      torch::stable::Tensor&       D_sf,
                                      torch::stable::Tensor const& A,
                                      torch::stable::Tensor const& B,
                                      torch::stable::Tensor const& global_scale,
                                      bool is_sf_swizzled_layout);

void fusedQuantizeMxAbsMax_host_sm100(torch::stable::Tensor&       D,
                                      torch::stable::Tensor&       D_sf,
                                      torch::stable::Tensor const& A,
                                      torch::stable::Tensor const& B,
                                      torch::stable::Tensor const& global_scale,
                                      bool is_sf_swizzled_layout);

void fusedQuantizeNvAbsMax_host_sm100(torch::stable::Tensor&       D,
                                      torch::stable::Tensor&       D_sf,
                                      torch::stable::Tensor const& A,
                                      torch::stable::Tensor const& B,
                                      torch::stable::Tensor const& global_scale,
                                      bool is_sf_swizzled_layout);

}  // namespace QUTLASS
