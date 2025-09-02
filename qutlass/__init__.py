#
# Copyright (C) 2025 Roberto L. Castro (Roberto.LopezCastro@ist.ac.at). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import qutlass._CUDA
from typing import Literal

def matmul_mxf4_bf16_tn(a: torch.Tensor,
                        b: torch.Tensor,
                        a_sf: torch.Tensor,
                        b_sf: torch.Tensor,
                        alpha: torch.Tensor) -> torch.Tensor:
    return qutlass._CUDA.matmul_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)

def matmul_ada_mxf4_bf16_tn(a: torch.Tensor,
                            b: torch.Tensor,
                            a_sf: torch.Tensor,
                            b_sf: torch.Tensor,
                            alpha: torch.Tensor) -> torch.Tensor:
    return qutlass._CUDA.matmul_ada_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)

def matmul_nvf4_bf16_tn(a: torch.Tensor,
                        b: torch.Tensor,
                        a_sf: torch.Tensor,
                        b_sf: torch.Tensor,
                        alpha: torch.Tensor) -> torch.Tensor:
    return qutlass._CUDA.matmul_nvf4_bf16_tn(a, b, a_sf, b_sf, alpha)

QuantMethod = Literal["quest", "abs_max"]
def ceil_div(a, b):
    return (a + b - 1) // b

def fusedQuantizeMx(a: torch.Tensor,
                    b: torch.Tensor,
                    #TODO: add global_scale for consistency?
                    *,
                    method: QuantMethod = "quest") -> tuple[torch.Tensor, torch.Tensor]:
    xh_e2m1 = torch.empty(*a.shape[:-1], a.size(-1) // 2,  dtype=torch.uint8, device=a.device)

    rows, cols = a.numel()//a.size(-1), a.size(-1)//32
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows  = n_row_blocks * 128
    padded_cols  = n_col_blocks * 4
    xh_e8m0      = torch.empty(padded_rows, padded_cols, dtype=torch.float8_e8m0fnu, device=a.device)

    if method=="quest":
        return qutlass._CUDA.fusedQuantizeMxQuest(a, b, xh_e2m1, xh_e8m0)
    elif method=="abs_max":
        return qutlass._CUDA.fusedQuantizeMxAbsMax(a, b, xh_e2m1, xh_e8m0)
    else:
        raise ValueError(f"invalid method {method!r}, "
                         "must be 'quest' or 'abs_max'")

def fusedQuantizeNv(a: torch.Tensor,
                    b: torch.Tensor,
                    global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xh_e2m1   = torch.empty(*a.shape[:-1], a.size(-1) // 2,  dtype=torch.uint8, device=a.device)

    rows, cols = a.numel()//a.size(-1), a.size(-1)//16
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows  = n_row_blocks * 128
    padded_cols  = n_col_blocks * 4
    xh_e4m3      = torch.empty(padded_rows, padded_cols, dtype=torch.float8_e4m3fn, device=a.device)

    return qutlass._CUDA.fusedQuantizeNv(a, b, xh_e2m1, xh_e4m3, global_scale)