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

void fusedQuantizeMxQuest_host(torch::Tensor&       D,
                               torch::Tensor&       D_sf,
                               torch::Tensor const& A,
                               torch::Tensor const& B);

void fusedQuantizeMxQuestWithMask_host(torch::Tensor&       D,
                                       torch::Tensor&       D_sf,
                                       torch::Tensor&       D_mask,
                                       torch::Tensor const& A,
                                       torch::Tensor const& B);

void fusedQuantizeMxAbsMax_host(torch::Tensor&       D,
                                torch::Tensor&       D_sf,
                                torch::Tensor const& A,
                                torch::Tensor const& B);

void fusedQuantizeMxQuestHad64_host(torch::Tensor&       D,
                                    torch::Tensor&       D_sf,
                                    torch::Tensor const& A,
                                    torch::Tensor const& B);

void fusedQuantizeMxAbsMaxHad64_host(torch::Tensor&       D,
                                     torch::Tensor&       D_sf,
                                     torch::Tensor const& A,
                                     torch::Tensor const& B);

void fusedQuantizeMxQuestHad128_host(torch::Tensor&       D,
                                     torch::Tensor&       D_sf,
                                     torch::Tensor const& A,
                                     torch::Tensor const& B);

void fusedQuantizeMxAbsMaxHad128_host(torch::Tensor&       D,
                                      torch::Tensor&       D_sf,
                                      torch::Tensor const& A,
                                      torch::Tensor const& B);

void fusedQuantizeMxQuestHad256_host(torch::Tensor&       D,
                                     torch::Tensor&       D_sf,
                                     torch::Tensor const& A,
                                     torch::Tensor const& B);

void fusedQuantizeMxAbsMaxHad256_host(torch::Tensor&       D,
                                      torch::Tensor&       D_sf,
                                      torch::Tensor const& A,
                                      torch::Tensor const& B);

void fusedQuantizeNvQuest_host(torch::Tensor&       D,
                               torch::Tensor&       D_sf,
                               torch::Tensor const& A,
                               torch::Tensor const& B,
                               torch::Tensor const& global_scale);

void fusedQuantizeNvQuestHad32_host(torch::Tensor&       D,
                                    torch::Tensor&       D_sf,
                                    torch::Tensor const& A,
                                    torch::Tensor const& B,
                                    torch::Tensor const& global_scale);

void fusedQuantizeNvQuestHad64_host(torch::Tensor&       D,
                                    torch::Tensor&       D_sf,
                                    torch::Tensor const& A,
                                    torch::Tensor const& B,
                                    torch::Tensor const& global_scale);

void fusedQuantizeNvQuestHad128_host(torch::Tensor&       D,
                                     torch::Tensor&       D_sf,
                                     torch::Tensor const& A,
                                     torch::Tensor const& B,
                                     torch::Tensor const& global_scale);

void fusedQuantizeNvAbsMax_host(torch::Tensor&       D,
                                torch::Tensor&       D_sf,
                                torch::Tensor const& A,
                                torch::Tensor const& B,
                                torch::Tensor const& global_scale);

void fusedQuantizeNvAbsMaxHad32_host(torch::Tensor&       D,
                                     torch::Tensor&       D_sf,
                                     torch::Tensor const& A,
                                     torch::Tensor const& B,
                                     torch::Tensor const& global_scale);

void fusedQuantizeNvAbsMaxHad64_host(torch::Tensor&       D,
                                     torch::Tensor&       D_sf,
                                     torch::Tensor const& A,
                                     torch::Tensor const& B,
                                     torch::Tensor const& global_scale);

void fusedQuantizeNvAbsMaxHad128_host(torch::Tensor&       D,
                                      torch::Tensor&       D_sf,
                                      torch::Tensor const& A,
                                      torch::Tensor const& B,
                                      torch::Tensor const& global_scale);

void fusedQuantizeMxAbsMax_host_sm100(torch::Tensor&       D,
                                      torch::Tensor&       D_sf,
                                      torch::Tensor const& A,
                                      torch::Tensor const& B,
                                      torch::Tensor const& global_scale);

void fusedQuantizeNvAbsMax_host_sm100(torch::Tensor&       D,
                                      torch::Tensor&       D_sf,
                                      torch::Tensor const& A,
                                      torch::Tensor const& B,
                                      torch::Tensor const& global_scale);

// v2 API: Arbitrary K support for SM100 (B200/B300)
void fusedQuantizeMxQuest_host_sm100_v2(torch::Tensor&       D,
                                         torch::Tensor&       D_sf,
                                         torch::Tensor const& A,
                                         torch::Tensor const& H,
                                         torch::Tensor const& global_scale);

void fusedQuantizeMxAbsMax_host_sm100_v2(torch::Tensor&       D,
                                          torch::Tensor&       D_sf,
                                          torch::Tensor const& A,
                                          torch::Tensor const& H,
                                          torch::Tensor const& global_scale);

}  // namespace QUTLASS
