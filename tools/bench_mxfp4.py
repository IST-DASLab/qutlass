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

import sys
import numpy as np
import torch
import triton

import qutlass
from qutlass import matmul_mxf4_bf16_tn, fusedQuantizeMx
from qutlass.utils import to_blocked

from fast_hadamard_transform import hadamard_transform

PROVIDER_CFGS = {
    "torch-bf16": dict(enabled=True),
    "mxfp4": dict(no_a_quant=False, enabled=True),
    "mxfp4-noquant": dict(no_a_quant=True, enabled=True),
}

_enabled = [k for k, v in PROVIDER_CFGS.items() if v["enabled"]]

def _quant_weight_mxfp4(b: torch.Tensor, forward_hadamard_matrix: torch.Tensor, device: str):
    weight_hf_e2m1, weight_hf_e8m0 = fusedQuantizeMx(b, forward_hadamard_matrix)
    weight_hf_scale_block = to_blocked(weight_hf_e8m0)
    return weight_hf_e2m1, weight_hf_scale_block


def build_mxfp4_runner(cfg, a, b, dtype, device):
    forward_hadamard_matrix = hadamard_transform(torch.eye(32, dtype=torch.bfloat16, device=torch.device('cuda')), scale=32 ** -.5)
    weight_hf_e2m1, weight_hf_scale_block = _quant_weight_mxfp4(b, forward_hadamard_matrix, device)

    if cfg["no_a_quant"]:
        # Pre-quantize activation
        input_hf_e2m1, input_hf_e8m0 = fusedQuantizeMx(a, forward_hadamard_matrix)
        input_hf_scale_block = to_blocked(input_hf_e8m0)

        def run():
            return matmul_mxf4_bf16_tn(
                input_hf_e2m1, weight_hf_e2m1, input_hf_scale_block, weight_hf_scale_block, 1.
            )

        return run

    # Quantize activation on-the-fly
    def run():
        input_hf_e2m1, input_hf_e8m0 = fusedQuantizeMx(a, forward_hadamard_matrix)
        input_hf_scale_block = to_blocked(input_hf_e8m0)
        return qutlass.matmul_mxf4_bf16_tn(
                input_hf_e2m1, weight_hf_e2m1, input_hf_scale_block, weight_hf_scale_block, 1.
        )

    return run

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        x_log=False,
        line_arg="provider",
        line_vals=_enabled,
        line_names=_enabled,
        ylabel="TFLOP/s (larger is better)",
        plot_name="BF16 vs MXFP4 GEMMs",
        args={},
    )
)
def benchmark(batch_size, provider, N, K):
    M = batch_size
    device = "cuda"
    dtype = torch.bfloat16

    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((N, K), device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch-bf16":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.linear(a, b),  warmup=25, rep=200, quantiles=quantiles
        )
    else:
        cfg = PROVIDER_CFGS[provider]
        run_quant = build_mxfp4_runner(cfg, a, b, dtype, device)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_quant(),  warmup=25, rep=200, quantiles=quantiles
        )

    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


MODELS = {
    'Llama7B': [
        (4096, 3 * 4096),
        (4096, 4096),
        (4096, 2 * 10752),
        (10752, 4096)
    ],
    'Llama13B': [
        (5120, 3 * 5120),
        (5120, 5120),
        (5120, 2 * 13568),
        (13568, 5120)
    ],
    'Llama33B': [
        (6656, 3 * 6656),
        (6656, 6656),
        (6656, 2 * 17664),
        (17664, 6656)
    ],
    'Llama65B': [
        (8192, 3 * 8192),
        (8192, 8192),
        (8192, 2 * 21760),
        (21760, 8192)
    ],
    #'Qwen3-0.6B': ((1024, 2048), (1024, 1024), (1024, 6144), (3072, 1024)),
    #'Qwen3-1.7B': ((2048, 4096), (2048, 2048), (2048, 12288), (6144, 2048)),
    #'Qwen3-4B': ((2560, 2560), (2560, 2560), (2560, 19456), (9728, 2560)),
    #'Qwen3-8B': ((4096, 4096), (4096, 4096), (4096, 24576), (12288, 4096)),
    #'Qwen3-14B': [(5120, 5120), (5120, 5120), (5120, 34816), (17408, 5120)],
    'Qwen3-32B': [(5120, 5120), (5120, 5120), (5120, 51200), (25600, 5120)],
    #'Llama-3.1-70B': [(8192, 8192), (8192, 8192), (8192, 57344), (28672, 8192)]
}

for model, layers in MODELS.items():
    for N, K in layers:
        print(f"{model}, N={N} K={K}, BF16 vs MXFP4 GEMMs TFLOP/s:")
        benchmark.run(
            print_data=True,
            show_plots=True,
            save_path=f"bench_mxfp4_res_n{N}_k{K}",
            N=N,
            K=K,
        )