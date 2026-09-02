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

import numpy as np
import pytest
import torch
from scipy.linalg import hadamard

from qutlass import matmul_nvf4_bf16_tn, fusedQuantizeNv
from qutlass.utils import to_blocked

try:
    from flashinfer import autotune

    _HAS_FLASHINFER = True
except Exception:
    _HAS_FLASHINFER = False

BACKENDS = ["cutlass"] + (["flashinfer"] if _HAS_FLASHINFER else [])

if not torch.cuda.is_available():
    pytest.skip("CUDA required for these tests.", allow_module_level=True)

def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5, dtype=dtype, device=device
    )

def _rtne_fp4(x: torch.Tensor):
    device = x.device
    grid = torch.tensor(
        [
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ],
        dtype=x.dtype,
        device=x.device,
    )
    grid_int = torch.tensor(
        [-1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.uint8,
        device=device,
    )
    inds = torch.bucketize(x, grid)
    lo, hi = (inds - 1).clamp(min=0, max=15), inds.clamp(min=0, max=15)
    g_lo, g_hi = grid[lo], grid[hi]
    pick_hi = (g_hi - x < x - g_lo) | (g_hi - x == x - g_lo) & (grid_int[hi] % 2 == 0)
    y = torch.where(pick_hi, g_hi, g_lo)
    y_int = torch.where(pick_hi, grid_int[hi], grid_int[lo])
    y_int_packed = (y_int[..., 1::2] & 0xF) << 4 | y_int[..., ::2] & 0xF
    return y, y_int_packed


def _dq_fp4(x_e2m1: torch.Tensor, x_e4m3: torch.Tensor, alpha: float):
    device = x_e2m1.device

    x_e2m1_i32 = x_e2m1.view(dtype=torch.uint8).to(dtype=torch.int32)
    x_e2m1_unpacked = torch.stack(
        [x_e2m1_i32 & 0xF, (x_e2m1_i32 >> 4) & 0xF], dim=-1
    ).flatten(start_dim=-2)

    grid_dq = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float64,
        device=device,
    )
    x_fp4_dq = grid_dq[x_e2m1_unpacked]

    scales_dq = x_e4m3.to(torch.float64)
    x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 16)) * scales_dq[..., None]).flatten(
        start_dim=-2
    ) / alpha  # * (4. / 3.)
    return x_dq, x_fp4_dq, scales_dq


def _unpack_mask(clip_mask: torch.Tensor) -> torch.Tensor:
    clip_mask_unpacked_dq = torch.zeros(
        *clip_mask.shape[:-1],
        clip_mask.size(-1) * 8,
        dtype=torch.bool,
        device=clip_mask.device,
    )
    for i in range(8):
        clip_mask_unpacked_dq[..., i::8] = (clip_mask >> i) & 1
    return clip_mask_unpacked_dq


def _forward_quantize_ref(
    x: torch.Tensor,
    h: torch.Tensor,
    rot_size: int,
    global_scale: float = 6.0,
):
    device = x.device

    xh_ref64 = (
        x.unflatten(dim=-1, sizes=(-1, rot_size)).to(dtype=torch.float64)
        @ h.reshape(rot_size, rot_size).to(dtype=torch.float64)
    ).flatten(start_dim=-2)

    abs_max = xh_ref64.unflatten(dim=-1, sizes=(-1, 16)).abs().amax(dim=-1)
    scales_ref64_ = global_scale * (abs_max / 6.0) + 1e-8

    xh_e4m3_ref = scales_ref64_.to(dtype=torch.float8_e4m3fn)
    scales_ref64 = xh_e4m3_ref.to(dtype=torch.float64)
    xh_scaled_ref64 = (
        xh_ref64.unflatten(dim=-1, sizes=(-1, 16)) / scales_ref64[..., None]
    ).flatten(start_dim=-2)

    xh_scaled_ref64 *= global_scale

    clip_mask_unpacked_ref = xh_scaled_ref64.abs() < 6.0
    clip_mask_ref = torch.zeros(
        *x.shape[:-1], x.size(-1) // 8, dtype=torch.uint8, device=device
    )
    for i in range(8):
        clip_mask_ref |= clip_mask_unpacked_ref[..., i::8].to(dtype=torch.uint8) << i

    xh_fp4_ref, xh_e2m1_ref = _rtne_fp4(xh_scaled_ref64)
    xh_dq, xh_fp4_dq, scales_dq = _dq_fp4(
        xh_e2m1_ref, xh_e4m3_ref, global_scale
    )
    clip_mask_unpacked_dq = _unpack_mask(clip_mask_ref)

    assert xh_fp4_dq.equal(xh_fp4_ref)
    assert scales_dq.equal(scales_ref64)
    assert clip_mask_unpacked_dq.equal(clip_mask_unpacked_ref)

    return (
        xh_dq,
        clip_mask_unpacked_ref,
        (xh_e2m1_ref, xh_e4m3_ref, clip_mask_ref),
    )


DTYPE = torch.bfloat16
DEVICE = torch.device("cuda:0")
ROT_SIZES = [16, 32, 64, 128]
GLOBAL_SCALES = [6.0]

LLAMA_MODELS = {
    "7B": [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
    "13B": [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
    "33B": [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
    "70B": [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
}

def _from_blocked(blocked_flat, rows, cols):
    """Inverse of to_blocked: blocked 1D sf -> row-major 2D."""
    n_col_blocks = (cols + 3) // 4
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = n_col_blocks * 4

    r = torch.arange(padded_rows, device=blocked_flat.device)
    c = torch.arange(padded_cols, device=blocked_flat.device)
    rr, cc = torch.meshgrid(r, c, indexing="ij")

    offsets = ((rr // 128) * (n_col_blocks * 512)
              + (cc // 4) * 512
              + (rr % 32) * 16
              + ((rr % 128) // 32) * 4
              + cc % 4)

    result = blocked_flat.view(torch.uint8)[offsets.reshape(-1).long()].view(
        blocked_flat.dtype).reshape(padded_rows, padded_cols)
    return result[:rows, :cols]


@pytest.fixture(autouse=True)
def _seed_each_test():
    np.random.seed(0)
    torch.random.manual_seed(0)

@pytest.mark.parametrize("rot_size", ROT_SIZES)
@pytest.mark.parametrize("global_scale_value", GLOBAL_SCALES)
@torch.inference_mode()
def test_fused_quantization(rot_size: int, global_scale_value: float):
    dtype, device = DTYPE, DEVICE
    h = get_hadamard_matrix(rot_size, dtype, device)
    global_scale = torch.tensor([global_scale_value], device=device)

    m, n, k = 504, 4096 * 2, 4096
    a = torch.randn(m, k, dtype=dtype, device=device) * 25.0
    b = torch.randn(n, k, dtype=dtype, device=device) * 25.0

    a_e2m1, a_sf = fusedQuantizeNv(a, h, global_scale, is_sf_swizzled_layout=True)
    b_e2m1, b_sf = fusedQuantizeNv(b, h, global_scale, is_sf_swizzled_layout=True)

    a_dq, *_ = _dq_fp4(a_e2m1, _from_blocked(a_sf, m, k // 16), alpha=1.0)
    b_dq, *_ = _dq_fp4(b_e2m1, _from_blocked(b_sf, n, k // 16), alpha=1.0)
    out_ref = a_dq @ b_dq.transpose(-2, -1)

    alpha = torch.tensor([1.0], device=device)
    out = matmul_nvf4_bf16_tn(
        a_e2m1, b_e2m1,
        a_sf.view(-1, k // 16), b_sf.view(-1, k // 16),
        alpha)
    assert out.equal(out_ref.to(dtype=out.dtype))


@pytest.mark.parametrize("model", list(LLAMA_MODELS.keys()))
@pytest.mark.parametrize("layer_idx", [0, 1, 2, 3])
@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("rot_size", ROT_SIZES)
@pytest.mark.parametrize("backend", BACKENDS)
@torch.inference_mode()
def test_llama_shapes(model: str, layer_idx: int, batch: int, rot_size: int, backend: str):
    dtype, device = DTYPE, DEVICE
    m = batch
    k, n = LLAMA_MODELS[model][layer_idx]

    h = get_hadamard_matrix(rot_size, dtype, device)

    a = torch.randn(m, k, dtype=dtype, device=device) * 25.0
    b = torch.randn(n, k, dtype=dtype, device=device) * 25.0

    global_scale = torch.tensor([1.0], device=device)

    a_e2m1, a_sf = fusedQuantizeNv(a, h, global_scale, is_sf_swizzled_layout=True)
    b_e2m1, b_sf = fusedQuantizeNv(b, h, global_scale, is_sf_swizzled_layout=True)

    a_dq, *_ = _dq_fp4(a_e2m1, _from_blocked(a_sf, m, k // 16), alpha=1.0)
    b_dq, *_ = _dq_fp4(b_e2m1, _from_blocked(b_sf, n, k // 16), alpha=1.0)
    out_ref = a_dq @ b_dq.transpose(-2, -1)

    alpha = torch.tensor([1.0], device=device)
    out = matmul_nvf4_bf16_tn(
        a_e2m1, b_e2m1,
        a_sf.view(-1, k // 16), b_sf.view(-1, k // 16),
        alpha, backend=backend)
    assert out.equal(out_ref.to(dtype=out.dtype))


@pytest.mark.parametrize("rot_size", ROT_SIZES)
@pytest.mark.parametrize("method", ["quest", "abs_max"])
@torch.inference_mode()
def test_row_major_sf_compatibility(rot_size: int, method: str):
    """The legacy row-major request round-trips to the direct blocked layout."""
    rows, groups_per_row = 129, 3  # exercise row and SF-column padding
    h = get_hadamard_matrix(rot_size, DTYPE, DEVICE)
    a = torch.randn(rows, rot_size * groups_per_row, dtype=DTYPE, device=DEVICE)
    global_scale = torch.tensor([1.0], device=DEVICE)

    _, sf_row_major = fusedQuantizeNv(
        a, h, global_scale, method=method, is_sf_swizzled_layout=False
    )
    _, sf_blocked = fusedQuantizeNv(
        a, h, global_scale, method=method, is_sf_swizzled_layout=True
    )

    assert to_blocked(sf_row_major, use_triton_kernel=True).view(torch.uint8).equal(
        sf_blocked.view(torch.uint8)
    )


@pytest.mark.parametrize("rows", [1, 127, 128, 129, 255, 256, 257])
@pytest.mark.parametrize("global_scale_value", [1.0, 6.0])
@pytest.mark.parametrize("rot_size", ROT_SIZES)
@pytest.mark.parametrize("groups_per_row", [1, 3])
@torch.inference_mode()
def test_sf_swizzled_layout_fusion(
    rows: int,
    global_scale_value: float,
    rot_size: int,
    groups_per_row: int,
):
    """Check NV blocked output, including row and SF-column padding."""
    k = rot_size * groups_per_row
    h = get_hadamard_matrix(rot_size, DTYPE, DEVICE)
    a = torch.randn(rows, k, dtype=DTYPE, device=DEVICE) * 10.0
    global_scale = torch.tensor([global_scale_value], device=DEVICE)

    fp4, sf = fusedQuantizeNv(a, h, global_scale)
    _, _, (fp4_ref, sf_ref_rm, _) = _forward_quantize_ref(
        a, h, rot_size, global_scale=global_scale_value
    )
    sf_ref_blocked = to_blocked(sf_ref_rm, use_triton_kernel=True)

    assert sf.view(torch.uint8).equal(sf_ref_blocked.view(torch.uint8))
