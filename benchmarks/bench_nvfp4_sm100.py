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

import os
import torch
import triton
from scipy.linalg import hadamard

import qutlass
from qutlass import matmul_nvf4_bf16_tn, fusedQuantizeNv
from qutlass.utils import to_blocked

try:
    from flashinfer import autotune

    _HAS_FLASHINFER = True
except Exception:
    _HAS_FLASHINFER = False


PROVIDER_CFGS = {
    "torch-bf16": dict(enabled=True),
    "nvfp4-cutlass": dict(backend="cutlass", no_a_quant=False, enabled=True),
    "nvfp4-cutlass-noquant": dict(backend="cutlass", no_a_quant=True, enabled=True),
    "nvfp4-flashinfer": dict(backend="flashinfer", no_a_quant=False, enabled=True),
    "nvfp4-flashinfer-noquant": dict(
        backend="flashinfer", no_a_quant=True, enabled=True
    ),
}

_enabled = [k for k, v in PROVIDER_CFGS.items() if v["enabled"]]


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5, dtype=dtype, device=device
    )


def _quant_weight_nvfp4(
    b: torch.Tensor,
    forward_hadamard_matrix: torch.Tensor,
    global_scale: torch.Tensor,
    device: str,
):
    weight_hf_e2m1, weight_hf_e4m3 = fusedQuantizeNv(
        b, forward_hadamard_matrix, global_scale
    )
    weight_hf_scale_block = to_blocked(weight_hf_e4m3, True)
    return weight_hf_e2m1, weight_hf_scale_block


def build_nvfp4_runner(cfg, a, b, forward_hadamard_matrix, dtype, device):
    alpha = torch.Tensor([1.0]).to("cuda")
    global_scale = torch.tensor([1.0]).to("cuda")
    weight_hf_e2m1, weight_hf_scale_block = _quant_weight_nvfp4(
        b, forward_hadamard_matrix, global_scale, device
    )

    if cfg["no_a_quant"] and cfg["backend"] == "cutlass":
        # Pre-quantize activation
        input_hf_e2m1, input_hf_e4m3 = fusedQuantizeNv(
            a, forward_hadamard_matrix, global_scale
        )
        input_hf_scale_block = to_blocked(input_hf_e4m3, True)

        def run():
            return matmul_nvf4_bf16_tn(
                input_hf_e2m1,
                weight_hf_e2m1,
                input_hf_scale_block,
                weight_hf_scale_block,
                alpha,
            )

        return run

    elif not cfg["no_a_quant"] and cfg["backend"] == "cutlass":
        # Quantize activation on-the-fly
        def run():
            input_hf_e2m1, input_hf_e4m3 = fusedQuantizeNv(
                a, forward_hadamard_matrix, global_scale
            )
            input_hf_scale_block = to_blocked(input_hf_e4m3, True)
            return matmul_nvf4_bf16_tn(
                input_hf_e2m1,
                weight_hf_e2m1,
                input_hf_scale_block,
                weight_hf_scale_block,
                alpha,
            )

        return run

    elif cfg["no_a_quant"] and cfg["backend"] == "flashinfer" and _HAS_FLASHINFER:
        # Pre-quantize activation
        input_hf_e2m1, input_hf_e4m3 = fusedQuantizeNv(
            a, forward_hadamard_matrix, global_scale
        )
        input_hf_scale_block = to_blocked(input_hf_e4m3, True)

        with autotune(True):
            matmul_nvf4_bf16_tn(
                input_hf_e2m1,
                weight_hf_e2m1,
                input_hf_scale_block,
                weight_hf_scale_block,
                alpha,
                backend="flashinfer",
            )
        torch.cuda.synchronize()

        def run():
            return matmul_nvf4_bf16_tn(
                input_hf_e2m1,
                weight_hf_e2m1,
                input_hf_scale_block,
                weight_hf_scale_block,
                alpha,
                backend="flashinfer",
            )

        return run

    elif not cfg["no_a_quant"] and cfg["backend"] == "flashinfer" and _HAS_FLASHINFER:
        input_hf_e2m1_tmp, input_hf_e4m3_tmp = fusedQuantizeNv(
            a, forward_hadamard_matrix, global_scale
        )
        input_hf_scale_block_tmp = to_blocked(input_hf_e4m3_tmp, True)

        with autotune(True):
            matmul_nvf4_bf16_tn(
                input_hf_e2m1_tmp,
                weight_hf_e2m1,
                input_hf_scale_block_tmp,
                weight_hf_scale_block,
                alpha,
                backend="flashinfer",
            )
        torch.cuda.synchronize()

        # Quantize activation on-the-fly
        def run():
            input_hf_e2m1, input_hf_e4m3 = fusedQuantizeNv(
                a, forward_hadamard_matrix, global_scale
            )
            input_hf_scale_block = to_blocked(input_hf_e4m3, True)
            return qutlass.matmul_nvf4_bf16_tn(
                input_hf_e2m1,
                weight_hf_e2m1,
                input_hf_scale_block,
                weight_hf_scale_block,
                alpha,
                backend="flashinfer",
            )

        return run


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[
            1,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            24576,
            32768,
        ],
        x_log=False,
        line_arg="provider",
        line_vals=_enabled,
        line_names=_enabled,
        ylabel="TFLOP/s (larger is better)",
        plot_name="BF16 vs NVFP4 GEMMs",
        args={},
    )
)
def benchmark(batch_size, provider, N, K, had_size):
    M = batch_size
    device = "cuda"
    dtype = torch.bfloat16

    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((N, K), device=device, dtype=dtype)
    forward_hadamard_matrix = get_hadamard_matrix(had_size, dtype, device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch-bf16":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.linear(a, b), rep=200, quantiles=quantiles
        )
    else:
        cfg = PROVIDER_CFGS[provider]
        run_quant = build_nvfp4_runner(
            cfg, a, b, forward_hadamard_matrix, dtype, device
        )
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: run_quant(), rep=200, quantiles=quantiles
        )

    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


MODELS = {
    #'Llama7B': [
    #    (4096, 3 * 4096),
    #    (4096, 4096),
    #    (4096, 2 * 10752),
    #    (10752, 4096)
    # ],
    #'Llama13B': [
    #    (5120, 3 * 5120),
    #    (5120, 5120),
    #    (5120, 2 * 13568),
    #    (13568, 5120)
    # ],
    #'Llama33B': [
    #    (6656, 3 * 6656),
    #    (6656, 6656),
    #    (6656, 2 * 17664),
    #    (17664, 6656)
    # ],
    #'Llama65B': [
    #    (8192, 3 * 8192),
    #    (8192, 8192),
    #    (8192, 2 * 21760),
    #    (21760, 8192)
    # ],
    #'Qwen3-0.6B': ((1024, 2048), (1024, 1024), (1024, 6144), (3072, 1024)),
    #'Qwen3-1.7B': ((2048, 4096), (2048, 2048), (2048, 12288), (6144, 2048)),
    #'Qwen3-4B': ((2560, 2560), (2560, 2560), (2560, 19456), (9728, 2560)),
    #'Qwen3-8B': ((4096, 4096), (4096, 4096), (4096, 24576), (12288, 4096)),
    #'Qwen3-14B': [(5120, 5120), (5120, 5120), (5120, 34816), (17408, 5120)],
    #'Qwen3-32B': [(5120, 5120), (5120, 51200), (25600, 5120)],
    #'Llama-3.1-70B': [(8192, 8192), (8192, 57344), (28672, 8192)],
    "Llama-3.1-70B": [(8192, 57344)]
}

for model, layers in MODELS.items():
    for K, N in layers:
        for had_size in [16, 32, 64, 128]:
            print(f"{model}, N={N} K={K}, HAD={had_size}, BF16 vs NVFP4 GEMMs TFLOP/s:")
            save_path = f"benchmarks_output/bench_nvfp4_res_n{N}_k{K}_had{had_size}_sm100"
            os.makedirs(save_path, exist_ok=True)
            benchmark.run(
                print_data=True,
                show_plots=True,
                save_path=save_path,
                N=N,
                K=K,
                had_size=had_size,
            )
