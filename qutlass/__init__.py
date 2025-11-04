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
from qutlass.utils import get_padded_shape_mx, get_padded_shape_nv, pad_to_block
from typing import Literal

import warnings

try:
    from flashinfer import mm_fp4

    _HAS_FLASHINFER = True
except Exception:
    _HAS_FLASHINFER = False


def matmul_mxf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    backend: Literal["cutlass", "flashinfer"] = "cutlass",
) -> torch.Tensor:
    if backend == "cutlass":
        return qutlass._CUDA.matmul_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)
    elif backend == "flashinfer":
        if not _HAS_FLASHINFER:
            raise ImportError(
                "flashinfer backend requested but not installed. "
                "Install with:\n"
                "  git clone https://github.com/flashinfer-ai/flashinfer.git --recursive\n"
                "  cd flashinfer && python -m pip install -v ."
            )

        m, packed_k = a.shape
        k = packed_k * 2
        n = b.shape[0]
        BLOCK = 32
        out = torch.empty([m, n], device=a.device, dtype=torch.bfloat16)

        mm_fp4(
            a,
            b.T,
            a_sf.view(torch.uint8).view(-1, k // BLOCK),
            b_sf.view(torch.uint8).view(-1, k // BLOCK).T,
            alpha,
            torch.bfloat16,
            out,
            block_size=BLOCK,
            use_8x4_sf_layout=False,
            backend="cudnn",
            use_nvfp4=False,
        )

        return out

    else:
        raise ValueError(f"invalid backend {backend!r}; use 'cutlass' or 'flashinfer'")


def matmul_ada_mxf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return qutlass._CUDA.matmul_ada_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)


def matmul_nvf4_bf16_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    backend: Literal["cutlass", "flashinfer"] = "cutlass",
) -> torch.Tensor:
    if backend == "cutlass":
        return qutlass._CUDA.matmul_nvf4_bf16_tn(a, b, a_sf, b_sf, alpha)
    elif backend == "flashinfer":
        if not _HAS_FLASHINFER:
            raise ImportError(
                "flashinfer backend requested but not installed. "
                "Install with:\n"
                "  git clone https://github.com/flashinfer-ai/flashinfer.git --recursive\n"
                "  cd flashinfer && python -m pip install -v ."
            )

        m, packed_k = a.shape
        k = packed_k * 2
        n = b.shape[0]
        BLOCK = 16
        out = torch.empty([m, n], device=a.device, dtype=torch.bfloat16)

        mm_fp4(
            a,
            b.T,
            a_sf.view(-1, k // BLOCK),
            b_sf.view(-1, k // BLOCK).T,
            alpha,
            torch.bfloat16,
            out,
            block_size=BLOCK,
            use_8x4_sf_layout=False,
            backend="cudnn",
            use_nvfp4=True,
        )

        return out

    else:
        raise ValueError(f"invalid backend {backend!r}; use 'cutlass' or 'flashinfer'")


def matmul_mxf8_bf16_tn(a: torch.Tensor,
                        b: torch.Tensor,
                        block_scale_a: torch.Tensor,
                        block_scale_b: torch.Tensor,
                        alpha: torch.Tensor) -> torch.Tensor:
    return qutlass._CUDA.matmul_mxf8_bf16_tn(a, b, block_scale_a, block_scale_b, alpha)

def matmul_mxf8_bf16_nn(a: torch.Tensor,
                        b: torch.Tensor,
                        block_scale_a: torch.Tensor,
                        block_scale_b: torch.Tensor,
                        alpha: torch.Tensor) -> torch.Tensor:
    return qutlass._CUDA.matmul_mxf8_bf16_nn(a, b, block_scale_a, block_scale_b, alpha)


def fusedQuantizeMx(
    a: torch.Tensor,
    b: torch.Tensor,
    # TODO: add global_scale for consistency?
    *,
    method: Literal["quest", "abs_max"] = "quest",
    return_mask: bool = False,
):
    padded_rows, padded_cols = get_padded_shape_mx(a)
    xh_e2m1 = torch.empty(
        *a.shape[:-1], a.size(-1) // 2, dtype=torch.uint8, device=a.device
    )
    xh_e8m0 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e8m0fnu, device=a.device
    )

    if method == "quest":
        if return_mask:
            clip_mask = torch.empty(
                *a.shape[:-1], a.size(-1) // 8, dtype=torch.uint8, device=a.device
            )
            return qutlass._CUDA.fusedQuantizeMxQuestWithMask(
                a, b, xh_e2m1, xh_e8m0, clip_mask
            )
        else:
            return qutlass._CUDA.fusedQuantizeMxQuest(a, b, xh_e2m1, xh_e8m0)
    elif method == "abs_max":
        if return_mask:
            raise ValueError("return_mask is only supported for method 'quest'")
        return qutlass._CUDA.fusedQuantizeMxAbsMax(a, b, xh_e2m1, xh_e8m0)
    else:
        raise ValueError(f"invalid method {method!r}, must be 'quest' or 'abs_max'")


def fusedQuantizeNv(
    a: torch.Tensor,
    b: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    method: Literal["quest", "abs_max"] = "abs_max",
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_rows, padded_cols = get_padded_shape_nv(a)
    xh_e2m1 = torch.empty(
        *a.shape[:-1], a.size(-1) // 2, dtype=torch.uint8, device=a.device
    )
    xh_e4m3 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e4m3fn, device=a.device
    )

    if method == "quest":
        return qutlass._CUDA.fusedQuantizeNvQuest(a, b, xh_e2m1, xh_e4m3, global_scale)
    elif method == "abs_max":
        return qutlass._CUDA.fusedQuantizeNvAbsMax(a, b, xh_e2m1, xh_e4m3, global_scale)
    else:
        raise ValueError(f"invalid method {method!r}, must be 'quest' or 'abs_max'")


def backward_t_bf16(
    x: torch.Tensor,
    h: torch.Tensor,
    xh_e2m1: torch.Tensor = None,
    xh_e8m0: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if xh_e2m1 is None:
        xh_e2m1 = torch.empty(
            *x.shape[:-2],
            x.size(-1),
            x.size(-2) // 2,
            dtype=torch.float4_e2m1fn_x2,
            device=h.device,
        )
    if xh_e8m0 is None:
        xh_e8m0 = torch.empty(
            *x.shape[:-2],
            x.size(-1),
            x.size(-2) // 32,
            dtype=torch.float8_e8m0fnu,
            device=h.device,
        )

    assert (
        x.dtype == h.dtype == torch.bfloat16
        and xh_e2m1.dtype == torch.float4_e2m1fn_x2
        and xh_e8m0.dtype == torch.float8_e8m0fnu
    )
    assert (
        x.is_contiguous()
        and h.is_contiguous()
        and xh_e2m1.is_contiguous()
        and xh_e8m0.is_contiguous()
    )

    qutlass._CUDA.backward_t_bf16(x, h, xh_e2m1, xh_e8m0)

    return xh_e2m1, xh_e8m0


def backward_qt_bf16(
    x_e2m1: torch.Tensor,
    x_e8m0: torch.Tensor,
    h: torch.Tensor,
    alpha: torch.Tensor,
    xh_e2m1: torch.Tensor = None,
    xh_e8m0: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if xh_e2m1 is None:
        xh_e2m1 = torch.empty(
            *x_e2m1.shape[:-2],
            x_e2m1.size(-1) * 2,
            x_e2m1.size(-2) // 2,
            dtype=torch.float4_e2m1fn_x2,
            device=h.device,
        )
    if xh_e8m0 is None:
        xh_e8m0 = torch.empty(
            *x_e8m0.shape[:-2],
            x_e8m0.size(-1) * 32,
            x_e8m0.size(-2) // 32,
            dtype=torch.float8_e8m0fnu,
            device=h.device,
        )

    # assert h.dtype == torch.bfloat16 and x_e2m1.dtype == xh_e2m1.dtype == torch.float4_e2m1fn_x2 and x_e8m0.dtype == xh_e8m0.dtype == torch.float8_e8m0fnu
    assert (
        x_e2m1.is_contiguous()
        and x_e8m0.is_contiguous()
        and h.is_contiguous()
        and xh_e2m1.is_contiguous()
        and xh_e8m0.is_contiguous()
    )

    qutlass._CUDA.backward_qt_bf16(x_e2m1, x_e8m0, h, alpha, xh_e2m1, xh_e8m0)

    return xh_e2m1, xh_e8m0

def backward_bf16_square_double_mxfp8(x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_bf16.size(0) % 128 != 0:
        x_bf16 = pad_to_block(x_bf16, [0], 128)
    x_fp8 = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
    row_scales = torch.empty(x_bf16.shape[0], x_bf16.shape[1] // 32, device=x_bf16.device, dtype=torch.float8_e8m0fnu)
    column_scales = torch.empty(x_bf16.shape[1], x_bf16.shape[0] // 32, device=x_bf16.device, dtype=torch.float8_e8m0fnu)

    qutlass._CUDA.backward_bf16_square_double_mxfp8(x_bf16, x_fp8, row_scales, column_scales)

    return x_fp8, row_scales, column_scales

def mxfp4_transpose_mxfp8(x_fp4: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: padding in kernel
    # >>>>
    if x_fp4.size(0) % 256 != 0:
        m = x_fp4.shape[0]
        m_up128 = ((m - 1) // 256) * 256 + 256
        x_fp4 = pad_to_block(x_fp4, [0], 256)
        scales[m:m_up128] = 1.0
    # <<<<

    x_fp8 = torch.empty(x_fp4.shape[1] * 2, x_fp4.shape[0], device=x_fp4.device, dtype=torch.float8_e4m3fn)
    shared_exps = torch.empty(x_fp4.shape[1] * 2, x_fp4.shape[0] // 32, device=x_fp4.device, dtype=torch.float8_e8m0fnu)

    qutlass._CUDA.mxfp4_transpose_mxfp8(x_fp4, scales, x_fp8, shared_exps)

    return x_fp8, shared_exps