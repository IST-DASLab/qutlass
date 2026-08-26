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

from qutlass import matmul_mxf4_bf16_tn, fusedQuantizeMx
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


def _dq_fp4(x_e2m1: torch.Tensor, x_e8m0: torch.Tensor, alpha: float):
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
    scales_dq = x_e8m0.to(torch.float64)

    x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 32)) * scales_dq[..., None]).flatten(
        start_dim=-2
    ) / alpha
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
    quest: bool = True,
    global_scale: float = 3.0,
):
    device = x.device
    xh_ref64 = (
        x.unflatten(dim=-1, sizes=(-1, rot_size)).to(dtype=torch.float64)
        @ h.reshape(rot_size, rot_size).to(dtype=torch.float64)
    ).flatten(start_dim=-2)

    if quest:
        scales_ref64_ = (
            xh_ref64.unflatten(dim=-1, sizes=(-1, 32)).std(dim=-1, correction=0)
            * (2.92247856 / 6.0)
            + 1e-8
        )
    else:
        abs_max = xh_ref64.unflatten(dim=-1, sizes=(-1, 32)).abs().amax(dim=-1)
        scales_ref64_ = abs_max + 1e-8

    xh_e8m0_ref = scales_ref64_.log2().floor().exp2().to(dtype=torch.float8_e8m0fnu)
    scales_ref64 = xh_e8m0_ref.to(dtype=torch.float64)

    xh_scaled_ref64 = (
        xh_ref64.unflatten(dim=-1, sizes=(-1, 32)) / scales_ref64[..., None]
    ).flatten(start_dim=-2)
    if not quest:
        xh_scaled_ref64 *= global_scale

    clip_mask_unpacked_ref = xh_scaled_ref64.abs() < 6.0
    clip_mask_ref = torch.zeros(
        *x.shape[:-1], x.size(-1) // 8, dtype=torch.uint8, device=device
    )
    for i in range(8):
        clip_mask_ref |= clip_mask_unpacked_ref[..., i::8].to(dtype=torch.uint8) << i

    xh_fp4_ref, xh_e2m1_ref = _rtne_fp4(xh_scaled_ref64)
    xh_dq, xh_fp4_dq, scales_dq = _dq_fp4(
        xh_e2m1_ref,
        xh_e8m0_ref,
        alpha=1.0 if quest else global_scale,
    )
    clip_mask_unpacked_dq = _unpack_mask(clip_mask_ref)

    assert xh_fp4_dq.equal(xh_fp4_ref)
    assert scales_dq.equal(scales_ref64)
    assert clip_mask_unpacked_dq.equal(clip_mask_unpacked_ref)

    return (
        xh_dq,
        clip_mask_unpacked_ref,
        (xh_e2m1_ref, xh_e8m0_ref, clip_mask_ref),
    )


DTYPE = torch.bfloat16
DEVICE = torch.device("cuda:0")

ROT_SIZES = [32, 64, 128]
SEEDS = [0]
BATCHES = [1, 16]

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
@torch.inference_mode()
def test_fused_quantization_absmax(rot_size: int):
    dtype, device = DTYPE, DEVICE

    h = get_hadamard_matrix(rot_size, dtype, device)

    m, n, k = 1, 504, 4096
    a = torch.randn(m, k, dtype=dtype, device=device) * 25.0
    b = torch.randn(n, k, dtype=dtype, device=device) * 25.0

    a_e2m1, a_sf = fusedQuantizeMx(a, h, method="abs_max", is_sf_swizzled_layout=True)
    b_e2m1, b_sf = fusedQuantizeMx(b, h, method="abs_max", is_sf_swizzled_layout=True)

    a_dq, *_ = _dq_fp4(a_e2m1, _from_blocked(a_sf, m, k // 32), alpha=1.0)
    b_dq, *_ = _dq_fp4(b_e2m1, _from_blocked(b_sf, n, k // 32), alpha=1.0)
    out_ref = a_dq @ b_dq.transpose(-2, -1)

    alpha = torch.tensor([1.0], device=device)
    out = matmul_mxf4_bf16_tn(a_e2m1, b_e2m1, a_sf, b_sf, alpha)
    assert out.equal(out_ref.to(dtype=out.dtype))


@pytest.mark.parametrize("rot_size", ROT_SIZES)
@torch.inference_mode()
def test_fused_quantization_quest(rot_size: int):
    dtype, device = DTYPE, DEVICE
    h = get_hadamard_matrix(rot_size, dtype, device)

    m, n, k = 504, 504, 2048
    a = torch.randn(m, k, dtype=dtype, device=device) * 25.0
    b = torch.randn(n, k, dtype=dtype, device=device) * 25.0

    a_e2m1, a_sf = fusedQuantizeMx(a, h, method="quest", is_sf_swizzled_layout=True)
    b_e2m1, b_sf = fusedQuantizeMx(b, h, method="quest", is_sf_swizzled_layout=True)

    a_dq, *_ = _dq_fp4(a_e2m1, _from_blocked(a_sf, m, k // 32), alpha=1.0)
    b_dq, *_ = _dq_fp4(b_e2m1, _from_blocked(b_sf, n, k // 32), alpha=1.0)
    out_ref = a_dq @ b_dq.transpose(-2, -1)

    alpha = torch.tensor([1.0], device=device)
    out = matmul_mxf4_bf16_tn(a_e2m1, b_e2m1, a_sf, b_sf, alpha)
    assert out.equal(out_ref.to(dtype=out.dtype))


@pytest.mark.parametrize("model", list(LLAMA_MODELS.keys()))
@pytest.mark.parametrize("layer_idx", [0, 1, 2, 3])
@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("had_size", ROT_SIZES)
@pytest.mark.parametrize("backend", BACKENDS)
@torch.inference_mode()
def test_llama_shapes(model: str, layer_idx: int, batch: int, had_size: int, backend: str):
    dtype, device = DTYPE, DEVICE
    m = batch
    k, n = LLAMA_MODELS[model][layer_idx]

    h = get_hadamard_matrix(had_size, dtype, device)

    a = torch.rand(m, k, dtype=dtype, device=device) * 25.0
    b = torch.rand(n, k, dtype=dtype, device=device) * 25.0

    a_e2m1, a_sf = fusedQuantizeMx(a, h, method="quest", is_sf_swizzled_layout=True)
    b_e2m1, b_sf = fusedQuantizeMx(b, h, method="quest", is_sf_swizzled_layout=True)

    a_dq, *_ = _dq_fp4(a_e2m1, _from_blocked(a_sf, m, k // 32), alpha=1.0)
    b_dq, *_ = _dq_fp4(b_e2m1, _from_blocked(b_sf, n, k // 32), alpha=1.0)
    out_ref = a_dq @ b_dq.transpose(-2, -1)

    alpha = torch.tensor([1.0], device=device)
    out = matmul_mxf4_bf16_tn(a_e2m1, b_e2m1, a_sf, b_sf, alpha, backend=backend)
    assert out.equal(out_ref.to(dtype=out.dtype))

@pytest.mark.parametrize("rot_size", ROT_SIZES)
@torch.inference_mode()
def test_sf_swizzled_layout_fusion(rot_size: int):
    """Verify kernel-blocked sf == to_blocked(reference_row_major_sf).

    SM80/SM120 kernels use an explicit blocked address formula. The SM100
    epilogue maps its native SfKMajorAtom coordinates to the same data layout.

    Uses direct scale tensor comparison (not GEMM outputs) to avoid
    differences from FP4 quantization implementation details.
    """
    dtype, device = DTYPE, DEVICE
    h = get_hadamard_matrix(rot_size, dtype, device)

    for method in ["quest", "abs_max"]:
        for groups_per_row in [1, 3, 4]:
            k = rot_size * groups_per_row
            for rows in [1, 128, 129]:
                a = torch.randn(rows, k, dtype=dtype, device=device) * 10.0

                # Kernel blocked output (flat 1-D)
                _, sf_kernel_flat = fusedQuantizeMx(a, h, method=method)

                # Reference: independently compute row-major sf → apply to_blocked
                _, _, (_, sf_ref_rm, _) = _forward_quantize_ref(
                    a, h, rot_size, quest=(method == "quest"))
                sf_ref_blocked = to_blocked(sf_ref_rm, use_triton_kernel=True)

                assert sf_kernel_flat.view(torch.uint8).equal(
                    sf_ref_blocked.view(torch.uint8)
                ), (
                    f"kernel blocked sf != to_blocked(reference sf) for "
                    f"method={method!r}, rot_size={rot_size}, rows={rows}, "
                    f"groups_per_row={groups_per_row}"
                )


@pytest.mark.parametrize("global_scale_value", [1.0, 2.0, 3.0, 6.0])
@torch.inference_mode()
def test_absmax_global_scale(global_scale_value: float):
    """Exercise the SM100 global-scale input against an independent reference."""
    rot_size = 128
    h = get_hadamard_matrix(rot_size, DTYPE, DEVICE)
    a = torch.randn(129, rot_size * 4, dtype=DTYPE, device=DEVICE) * 10.0
    global_scale = torch.tensor([global_scale_value], device=DEVICE)

    fp4, sf = fusedQuantizeMx(
        a, h, method="abs_max", global_scale=global_scale
    )
    _, _, (fp4_ref, sf_ref_rm, _) = _forward_quantize_ref(
        a,
        h,
        rot_size,
        quest=False,
        global_scale=global_scale_value,
    )
    sf_ref_blocked = to_blocked(sf_ref_rm, use_triton_kernel=True)

    assert sf.view(torch.uint8).equal(sf_ref_blocked.view(torch.uint8))
    mismatch = (fp4.view(torch.uint8) != fp4_ref.view(torch.uint8)).float().mean()
    assert mismatch <= 1e-4


@torch.inference_mode()
def test_quest_mask_uses_general_blocked_layout():
    """Cover multiple SF column blocks and the 128-row padding boundary."""
    rot_size = 32
    h = get_hadamard_matrix(rot_size, DTYPE, DEVICE)
    a = torch.randn(129, 512, dtype=DTYPE, device=DEVICE) * 10.0

    fp4, sf, mask = fusedQuantizeMx(a, h, method="quest", return_mask=True)
    _, _, (fp4_ref, sf_ref_rm, mask_ref) = _forward_quantize_ref(
        a, h, rot_size, quest=True
    )
    sf_ref_blocked = to_blocked(sf_ref_rm, use_triton_kernel=True)

    assert sf.view(torch.uint8).equal(sf_ref_blocked.view(torch.uint8))
    assert mask.equal(mask_ref)
    mismatch = (fp4.view(torch.uint8) != fp4_ref.view(torch.uint8)).float().mean()
    assert mismatch <= 1e-4


@torch.inference_mode()
def test_row_major_sf_request_is_rejected():
    h = get_hadamard_matrix(32, DTYPE, DEVICE)
    a = torch.randn(1, 128, dtype=DTYPE, device=DEVICE)
    with pytest.raises(ValueError, match="row-major scale-factor"):
        fusedQuantizeMx(a, h, is_sf_swizzled_layout=False)


@torch.inference_mode()
def test_quest_rejects_global_scale():
    h = get_hadamard_matrix(32, DTYPE, DEVICE)
    a = torch.randn(1, 128, dtype=DTYPE, device=DEVICE)
    global_scale = torch.tensor([3.0], device=DEVICE)
    with pytest.raises(ValueError, match="only supported for method 'abs_max'"):
        fusedQuantizeMx(a, h, method="quest", global_scale=global_scale)
