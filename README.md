# QuTLASS v0.2.0
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Static Badge](https://img.shields.io/badge/CUTLASS-4.2.1-purple)](https://github.com/NVIDIA/cutlass)
[![Static Badge](https://img.shields.io/badge/PyTorch-2.8-red)](https://download.pytorch.org/whl/nightly/cu128)

**CUTLASS-powered quantized BLAS library for low-bit deep learning on NVIDIA Blackwell GPUs.**

QuTLASS is a high-performance library designed for **low-precision kernel support** in deep learning quantization, built on top of [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass).
It introduces **narrow-precision microscaling routines** tailored for **quantized Large Language Model (LLM)** inference and training on **NVIDIA Blackwell GPUs**.

[![arXiv](https://img.shields.io/badge/arXiv-2509.23202-b31b1b.svg)](https://arxiv.org/pdf/2509.23202)
[![arXiv](https://img.shields.io/badge/arXiv-2505.14669-b31b1b.svg)](https://arxiv.org/abs/2505.14669)

---

## Table of Contents
- [Microscaling in Blackwell](#microscaling-in-blackwell)
- [What’s New in v0.2](#-what-is-new-in-qutlass-v02)
- [Features from Previous Versions](#-features-from-previous-versions)
- [Getting Started](#️-getting-started)
- [Usage Examples](#-usage-example)
- [Benchmarks](#-benchmarks)
- [Citation](#-citation)

---

## Microscaling in Blackwell
The new **Blackwell** architecture supports native matrix multiplication with microscaling, using scale factors in the form:

$$
D = C + (A \times \mathrm{SFA}) \cdot (B \times \mathrm{SFB})
$$

Here, the scale factors are applied along the inner ($K$) dimension of the GEMM.
For MXFP types, one scale factor is shared by every 32 elements along $K$ (group size $gs=32$).
Thus, for an $M \times K$ matrix $A$, the corresponding scale matrix $\mathrm{SFA}$ has dimensions:

$$
M \times \left\lceil K / gs \right\rceil
$$

## 🚀 What is new in QuTLASS v0.2:
- **FlashInfer backend** support for **B200 GPUs**
- **Quantization-Aware Training (QAT)** via MXFP types:
  - Quartet clipping mask computation integrated in quantization routines
  - Prototype backward kernels for MXFP4 (`sm_120`) and MXFP8 (`sm_100`)
  - Integrated CUTLASS MXFP8 backward GEMM kernels (TN and NN layouts)
- **Updated Transformers Integration** for QAT ([#41897](https://github.com/huggingface/transformers/pull/41897))
- **Nanochat-QAT Integration** ([#1](https://github.com/IST-DASLab/nanochat-qat/pull/1))

## 🧩 Features from Previous Versions
### From QuTLASS v0.1:
- Support for ```sm_100``` GPUs (e.g., NVIDIA B200).
- NVFP4 Microscaling:
    - Full W4A4 quantization support.
    - Online rotations:
        - Fused transform + quantization + scale computation.
        - Rotation matrices loaded at runtime, allowing any transformation to be applied.
    - NVFP4 Matmul Kernels:
        - CUTLASS-backed NVFP4:NVFP4 with block-scale reordering.
    - Quantization:
        - Abs-Max supported.
- Multiple rotation sizes (16/32/64/128) supported for both MXFP4 and NVFP4.
- vLLM Integration ([PR #24440](https://github.com/vllm-project/vllm/pull/24440))

### From QuTLASS v0.0:
- MXFP4 microscaling support, with
- Weight and Activation quantization (W4A4)
- *Online rotations*: fused kernel for online transforms, quantization, and scale computation.
    - Transformations matching the microscaling group sizes (i.e., 32 for MXFP4).
    - Compatible with any rotation matrix defined (e.g., Identity, Hadamard, DCT), as they are loaded in runtime.
- Multiple quantization schemes:
    - [Quartet](https://arxiv.org/pdf/2505.14669) (i.e., [Quest-like](https://arxiv.org/abs/2502.05003)).
    - Abs-Max.
- Matmul kernels:
    - CUTLASS-backed MXFP4:MXFP4 kernel with block-scale reordering.
    - Prototype kernel for small batch sizes (no reordering required).
- Transformers Integration ([PR #38696](https://github.com/huggingface/transformers/pull/38696))

> **Note:** QuTLASS is under *active development* and not yet fully optimized.

## ⚙️ Getting Started

### Requirements:

- **NVIDIA Blackwell GPU** (Compute capabilities supported: `sm_120a` and `sm_100a`)
- **Compatible drivers**: CUDA 12.8 or newer

### Installation:

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Install QuTLASS (in editable mode):

```bash
pip install --no-build-isolation -e .
```

in the root folder of this repository.

> **Note:** To generate accurate quantized models using MXFP4 or NVFP4 formats, refer to the [FP-Quant repository](https://github.com/IST-DASLab/FP-Quant).

## 🧪 Usage example

Correctness tests can be executed via ```python tests/mxfp4_test.py``` and benchmarks via ```python benchmarks/bench_mxfp4.py```.

The fused quantization kernel can be invoked directly through ```qutlass.fusedQuantizeMx(a, h, method)```. Here, ```a``` is the input tensor to quantize, ```h``` is the Hadamard matrix, and ```method``` is the quantization scheme specified as ```Literal["quest", "abs_max"]```.
The kernel interface is defined in ```qutlass/csrc/fused_quantize_mx.cu```.
The outputs include ```aq```, the quantized data in FP4 (```e2m1```), and ```a_sf``` the corresponding scaling factors in FP8 (```e8m0```).

The matmul kernel can be called via ```qutlass.matmul_mxf4_bf16_tn(aq, bq, a_sf, b_sf, alpha)```. Its implementation can be found in ```qutlass/csrc/gemm.cu```.
To use this matmul kernel, the scaling factors must be first rearranged into a block-scaled swizzle [format](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout).
The ```qutlass.to_blocked```, located in ```qutlass/utils.py```, handles this reordering.

In addition to the previous CUTLASS-powered MXFP4 matmul kernel, we provide a custom prototype kernel that can be called via ```qutlass.matmul_ada_mxf4_bf16_tn(...)```.
This implementation is located in ```qutlass/csrc/gemm_ada.cu``` and does **not** require the previous invocation to ```to_blocked```.
Optimization efforts for this kernel have primarily targeted small batch sizes(i.e., $bs=1\sim 32$). For larger sizes, ```qutlass.matmul_mxf4_bf16_tn``` is recommended.

This applies also to NVFP4, which is functionally equivalent aside from minor naming changes.

## 📈 Benchmarks

### Microbenchmarks
The following illustrate the performance of QuTLASS MXFP4 across various batch sizes.
Ideal performance refers to pure matrix multiplication in FP4, without any overhead from quantization.
Actual performance includes the full pipeline: Hadamard rotation, data quantization, scale computation, and block-scale reordering.

<div class="row"><center>
  <div class="column">
    <img src="assets/flops_mxfp4_sm120_cutlass.svg" alt="Llama-65B MXFP4:MXFP4" style="width:45%">
    <img src="assets/flops_nvfp4_sm120_cutlass.svg" alt="Llama-65B MXFP4:MXFP4" style="width:45%">
  </div>
  <em>QuTLASS performance on a single Qwen3-32B layer with NVIDIA RTX5090 GPU - CUTLASS backend</em>
 </center>
</div>
</br>
<div class="row"><center>
  <div class="column">
    <img src="assets/flops_mxfp4_sm100_flashinfer.svg" alt="Llama-70B MXFP4:MXFP4" style="width:45%">
    <img src="assets/flops_nvfp4_sm100_flashinfer.svg" alt="Llama-70B MXFP4:MXFP4" style="width:45%">
  </div>
  <em>QuTLASS performance on a single Llama-3.1-70B layer with NVIDIA B200 GPU - FlashInfer backend</em>
 </center>
</div>

### End-to-end Inference Speedups
The following results show the inference speedup of QuTLASS MXFP4 over PyTorch BF16 in Transformers, as a function of batch size and sequence length on 8B and 14B-parameter models.
MXFP4 delivers consistent performance gains across all batch sizes, with speedups increasing progressively and peaking at $\approx 4\times$ compared to BF16.

<div class="row"><center>
  <div class="column">
    <img src="assets/qwen3-8b-end-to-end-prefill-speedup-mxfp4-vs-bf16-on-rtx5090.svg" alt="Prefill MXFP4:MXFP4" style="width:90%">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="assets/qwen3-14b-end-to-end-prefill-speedup-mxfp4-vs-bf16-on-rtx5090.svg" alt="Prefill MXFP4:MXFP4" style="width:90%">
  </div>
 </center>
</div>

In order to generate recipes for efficient and accurate weight + activation quantization for low-bit MXFP and NVFP formats, please refer to [FP-Quant](https://github.com/IST-DASLab/FP-Quant).

### End-to-end Training Speedups
The following results show some QAT performance using QuTLASS.
Using our Transformers integration, an MXFP4:MXFP8 QAT scheme applied to Llama-3.1-8B recovers over half of the lost performance after only ~100M training tokens, while training 30% faster than BF16 pseudo-quantization QAT.

<div class="row"><center>
  <div class="column">
    <img src="assets/training.png" width="650"/>
  </div>
 <em>Llama-3.1-8B-Instruct</em>
 </center>
</div>

For efficient and accurate QAT recipes for low-bit MXFP formats, see [nanochat-qat](https://github.com/IST-DASLab/nanochat-qat/pull/1) and [FP-Quant](https://github.com/huggingface/transformers/pull/41897).

## 📚 Citation

```bibtex
@misc{qutlass2025,
      title={QuTLASS: CUTLASS-Powered Quantized BLAS for Deep Learning},
      author={Roberto L. Castro, and Dan Alistarh},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/IST-DASLab/qutlass}},
}
```
