#include "include/cuda_utils.h"

int64_t get_device_attribute(int64_t attribute, int64_t device_id) {
  // Return the cached value on subsequent calls
  static int value = [=]() {
    int device = static_cast<int>(device_id);
    if (device < 0) {
      CUDA_CHECK(cudaGetDevice(&device));
    }
    int value;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &value, static_cast<cudaDeviceAttr>(attribute), device));
    return static_cast<int>(value);
  }();

  return value;
}

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id) {
  int64_t attribute;
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
  // cudaDevAttrMaxSharedMemoryPerBlockOptin = 97 if not is_hip() else 74

#ifdef USE_ROCM
  attribute = hipDeviceAttributeMaxSharedMemoryPerBlock;
#else
  attribute = cudaDevAttrMaxSharedMemoryPerBlockOptin;
#endif

  return get_device_attribute(attribute, device_id);
}
