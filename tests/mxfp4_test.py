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

import unittest
import torch
import numpy as np

import qutlass
from qutlass import matmul_mxf4_bf16_tn, fusedQuantizeMx
from qutlass.utils import to_blocked
from fast_hadamard_transform import hadamard_transform

def _rtne_fp4(x: torch.Tensor):
    device = x.device
    grid = torch.tensor([-6., -4., -3., -2., -1.5, -1., -.5, -0., 0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=x.dtype, device=x.device)
    grid_int = torch.tensor([-1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device=device)
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
    x_e2m1_unpacked = torch.stack([x_e2m1_i32 & 0xF, (x_e2m1_i32 >> 4) & 0xF], dim=-1).flatten(start_dim=-2)

    grid_dq = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.,], dtype=torch.float64, device=device)
    x_fp4_dq = grid_dq[x_e2m1_unpacked]

    scales_dq = x_e8m0.to(torch.float64)
    x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 32)) * scales_dq[..., None]).flatten(start_dim=-2) / alpha  #* (4. / 3.)

    return x_dq, x_fp4_dq, scales_dq

def _unpack_mask(clip_mask: torch.Tensor) -> torch.Tensor:
    clip_mask_unpacked_dq = torch.zeros(*clip_mask.shape[:-1], clip_mask.size(-1) * 8, dtype=torch.bool, device=clip_mask.device)
    for i in range(8):
        clip_mask_unpacked_dq[..., i::8] = (clip_mask >> i) & 1
    return clip_mask_unpacked_dq

def _forward_quantize_ref(x: torch.Tensor, h: torch.Tensor, quest: bool=True):
    device = x.device

    xh_ref64 = (x.unflatten(dim=-1, sizes=(-1, 32)).to(dtype=torch.float64) @ h.reshape(32, 32).to(dtype=torch.float64)).flatten(start_dim=-2)
    if quest:
        scales_ref64_ = xh_ref64.unflatten(dim=-1, sizes=(-1, 32)).std(dim=-1, correction=0) * (2.92247856 / 6.) + 1e-8
    else:
        abs_max = xh_ref64.unflatten(dim=-1, sizes=(-1, 32)).abs().amax(dim=-1)
        scales_ref64_ = abs_max + 1e-8

    xh_e8m0_ref = scales_ref64_.log2().floor().exp2().to(dtype=torch.float8_e8m0fnu)
    scales_ref64 = xh_e8m0_ref.to(dtype=torch.float64)
    xh_scaled_ref64 = (xh_ref64.unflatten(dim=-1, sizes=(-1, 32)) / scales_ref64[..., None]).flatten(start_dim=-2)
    if not quest:
        xh_scaled_ref64 *= 3

    clip_mask_unpacked_ref = xh_scaled_ref64.abs() < 6.
    clip_mask_ref = torch.zeros(*x.shape[:-1], x.size(-1) // 8, dtype=torch.uint8, device=device)
    for i in range(8):
        clip_mask_ref |= clip_mask_unpacked_ref[..., i::8].to(dtype=torch.uint8) << i

    xh_fp4_ref, xh_e2m1_ref = _rtne_fp4(xh_scaled_ref64)

    xh_dq, xh_fp4_dq, scales_dq = _dq_fp4(xh_e2m1_ref, xh_e8m0_ref, alpha=1. if quest else 3.0)
    clip_mask_unpacked_dq = _unpack_mask(clip_mask_ref)
    assert xh_fp4_dq.equal(xh_fp4_ref)
    assert scales_dq.equal(scales_ref64)
    assert clip_mask_unpacked_dq.equal(clip_mask_unpacked_ref)

    return xh_dq, clip_mask_unpacked_ref, (xh_e2m1_ref, xh_e8m0_ref, clip_mask_ref)

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

dtype = torch.bfloat16
device = 'cuda:0'

hadamard_matrix = hadamard_transform(torch.eye(32, dtype=dtype, device=device), scale=32. ** -.5)

class Test(unittest.TestCase):
    def test_fused_quantization(self):
        hadamard_matrix = hadamard_transform(torch.eye(32, dtype=dtype, device=device), scale=32. ** -.5)
        x = torch.randn(2, 4096, 4096, dtype=dtype, device=device) * 25.

        ################# Abs-Max
        # Quantization kernel
        xh_dq_ref, clip_mask_unpacked_ref, (xh_e2m1_ref, xh_e8m0_ref, clip_mask_ref) = _forward_quantize_ref(x, hadamard_matrix, False)

        xh_e2m1, xh_e8m0 = fusedQuantizeMx(x, hadamard_matrix, method='abs_max')

        xh_e8m0 = xh_e8m0.reshape(2, 4096, 4096//32) #FIXME:
        print(xh_e2m1.shape, xh_e8m0.shape)
        xh_dq, *_ = _dq_fp4(xh_e2m1, xh_e8m0, alpha=3.0)
        torch.testing.assert_close(xh_dq, xh_dq_ref, rtol=0.34, atol=100)
        assert (xh_dq != xh_dq_ref).float().mean() <= 1e-4

        # Matmul kernel
        m, n, k = 1, 504, 4096
        a = torch.randn(m, k, dtype=dtype, device=device) * 25.
        b = torch.randn(n, k, dtype=dtype, device=device) * 25.

        a_e2m1, a_e8m0 = fusedQuantizeMx(a, hadamard_matrix, method='abs_max')
        b_e2m1, b_e8m0 = fusedQuantizeMx(b, hadamard_matrix, method='abs_max')

        a_dq, *_ = _dq_fp4(a_e2m1, a_e8m0[:m,:k], alpha=1.)
        b_dq, *_ = _dq_fp4(b_e2m1, b_e8m0[:n,:k], alpha=1.)
        out_ref = a_dq @ b_dq.transpose(-2, -1)

        a_scale_block = to_blocked(a_e8m0)
        b_scale_block = to_blocked(b_e8m0)
        out = matmul_mxf4_bf16_tn(a_e2m1, b_e2m1, a_scale_block, b_scale_block, 1.)
        #out = qutlass.matmul_ada_mxf4_bf16_tn(a_e2m1, b_e2m1, a_e8m0, b_e8m0, 1.0)

        assert out.equal(out_ref.to(dtype=out.dtype))

        ################# Quartet
        # Quantization kernel
        xh_dq_ref, clip_mask_unpacked_ref, (xh_e2m1_ref, xh_e8m0_ref, clip_mask_ref) = _forward_quantize_ref(x, hadamard_matrix, True)
        xh_e2m1, xh_e8m0 = fusedQuantizeMx(x, hadamard_matrix, method='quest')

        diff = (xh_e2m1.view(dtype=xh_e2m1_ref.dtype) != xh_e2m1_ref).sum().item()
        print(diff, diff / xh_e2m1.numel())
        xh_e8m0 = xh_e8m0.reshape(2, 4096, 4096//32) #FIXME:
        xh_dq, *_ = _dq_fp4(xh_e2m1, xh_e8m0, alpha=1.)

        assert xh_dq.equal(xh_dq_ref)

        # Matmul kernel
        #m, n, k = 4096 * 3, 4096 * 2, 4096
        m, n, k = 504, 504, 2048
        a = torch.randn(m, k, dtype=dtype, device=device) * 25.
        b = torch.randn(n, k, dtype=dtype, device=device) * 25.

        a_e2m1, a_e8m0 = fusedQuantizeMx(a, hadamard_matrix, method='quest')
        b_e2m1, b_e8m0 = fusedQuantizeMx(b, hadamard_matrix, method='quest')

        a_dq, *_ = _dq_fp4(a_e2m1, a_e8m0[:m,:k], alpha=1.)
        b_dq, *_ = _dq_fp4(b_e2m1, b_e8m0[:n, :k], alpha=1.)
        out_ref = a_dq @ b_dq.transpose(-2, -1)

        a_scale_block = to_blocked(a_e8m0, True)
        b_scale_block = to_blocked(b_e8m0, True)
        out = matmul_mxf4_bf16_tn(a_e2m1, b_e2m1, a_scale_block, b_scale_block, 1.)
        assert out.equal(out_ref.to(dtype=out.dtype))

    def run_problem(self, m, n, k):
        print(m, n, k)

        a = torch.rand(m, k, dtype=dtype, device=device) * 25.
        b = torch.rand(n, k, dtype=dtype, device=device) * 25.

        a_e2m1, a_e8m0 = fusedQuantizeMx(a, hadamard_matrix, method='quest')
        b_e2m1, b_e8m0 = fusedQuantizeMx(b, hadamard_matrix, method='quest')

        a_dq, *_ = _dq_fp4(a_e2m1, a_e8m0[:m,:k], alpha=1.)
        b_dq, *_ = _dq_fp4(b_e2m1, b_e8m0[:n, :k], alpha=1.)
        out_ref = a_dq @ b_dq.transpose(-2, -1)

        a_scale_block = to_blocked(a_e8m0, True)
        b_scale_block = to_blocked(b_e8m0, True)
        out = matmul_mxf4_bf16_tn(a_e2m1, b_e2m1, a_scale_block, b_scale_block, 1.)

        #out = qutlass.matmul_ada_mxf4_bf16_tn(a_e2m1, b_e2m1, a_e8m0, b_e8m0, 1.0)

        assert out.equal(out_ref.to(dtype=out.dtype))

    def run_problem_ada(self, m, n, k):
        print(m, n, k)

        a = torch.rand(m, k, dtype=dtype, device=device) * 25.
        b = torch.rand(n, k, dtype=dtype, device=device) * 25.

        a_e2m1, a_e8m0 = fusedQuantizeMx(a, hadamard_matrix, method='quest')
        b_e2m1, b_e8m0 = fusedQuantizeMx(b, hadamard_matrix, method='quest')

        a_dq, *_ = _dq_fp4(a_e2m1, a_e8m0[:m,:k], alpha=1.)
        b_dq, *_ = _dq_fp4(b_e2m1, b_e8m0[:n, :k], alpha=1.)
        out_ref = a_dq @ b_dq.transpose(-2, -1)

        out = qutlass.matmul_ada_mxf4_bf16_tn(a_e2m1, b_e2m1, a_e8m0, b_e8m0, 1.0)

        assert out.equal(out_ref.to(dtype=out.dtype))

    def test_llama_shapes(self):
        print()
        #return
        MODELS = {
            ' 7B': [
                (4096, 3 * 4096),
                (4096, 4096),
                (4096, 2 * 10752),
                (10752, 4096)
            ],
            '13B': [
                (5120, 3 * 5120),
                (5120, 5120),
                (5120, 2 * 13568),
                (13568, 5120)
            ],
            '33B': [
                (6656, 3 * 6656),
                (6656, 6656),
                (6656, 2 * 17664),
                (17664, 6656)
            ],
            '70B': [
                (8192, 3 * 8192),
                (8192, 8192),
                (8192, 2 * 21760),
                (21760, 8192)
            ]
        }
        for _, layers in MODELS.items():
            for layer in layers:
                for batch in [1, 16]:
                    self.run_problem(batch, layer[1], layer[0])
                    self.run_problem_ada(batch, layer[1], layer[0])

if __name__ == '__main__':
    unittest.main()
