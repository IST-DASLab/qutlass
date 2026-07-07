#pragma once

#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <climits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include "cutlass/cutlass.h"

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                              \
  {                                                        \
    cutlass::Status error = status;                        \
    STD_TORCH_CHECK(error == cutlass::Status::kSuccess,    \
                    cutlassGetStatusString(error));        \
  }

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                            \
  {                                                                   \
    cudaError_t error = status;                                       \
    STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error)); \
  }

inline cudaStream_t get_current_cuda_stream(int32_t device_index) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}

inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return max_shared_mem_per_block_opt_in;
}

int32_t get_sm_version_num();

/**
 * A wrapper for a kernel that is used to guard against compilation on
 * architectures that will never use the kernel. The purpose of this is to
 * reduce the size of the compiled binary.
 * __CUDA_ARCH__ is not defined in host code, so this lets us smuggle the ifdef
 * into code that will be executed on the device where it is defined.
 */
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};