#pragma once

#include <initializer_list>
#include <iostream>
#include <sstream>

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/util/Exception.h>

namespace {

// methods in here are meant to mirror some of the methods 
// in PyTorch's aten/src/ATen/TensorUtils.cpp file

using torch::stable::Tensor;
using torch::headeronly::DeviceType;

struct TensorArg {
  const Tensor& tensor;
  const char* name;
  int pos;
};

inline std::ostream& operator<<(std::ostream& out, const TensorArg& t) {
  if (t.pos == 0) {
    // 0 is distinguished; it usually indicates 'self' or the return
    // tensor
    out << '\'' << t.name << '\'';
  } else {
    out << "argument #" << t.pos << " '" << t.name << '\'';
  }
  return out;
}

inline const char* device_type_name(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return "cpu";
    case DeviceType::CUDA:
      return "cuda";
    case DeviceType::HIP:
      return "hip";
    case DeviceType::XLA:
      return "xla";
    case DeviceType::MPS:
      return "mps";
    case DeviceType::XPU:
      return "xpu";
    case DeviceType::Meta:
      return "meta";
    case DeviceType::PrivateUse1:
      return "privateuseone";
    default:
      return "unknown";
  }
}

inline void append_device_string(std::ostream& out, const Tensor& t) {
  const auto device = t.device();
  out << device_type_name(device.type());
  if (device.has_index()) {
    out << ':' << device.index();
  }
}

inline void check_all_contiguous(const char* op,
                                 std::initializer_list<TensorArg> args) {
  for (const auto& arg : args) {
    if (arg.tensor.is_contiguous()) {
      continue;
    }
    std::ostringstream oss;
    oss << "Expected contiguous tensor, but got non-contiguous tensor for " << arg
        << " (while checking arguments for " << op << ")";
    STD_TORCH_CHECK(false, oss.str());
  }
}

inline void check_device_type_cuda(const char* op,
                                   std::initializer_list<Tensor> tensors) {
  for (const auto& tensor : tensors) {
    if (tensor.device().is_cuda()) {
      continue;
    }
    std::ostringstream oss;
    oss << "Expected tensor to have cuda DeviceType, but got tensor with "
        << device_type_name(tensor.device().type())
        << " DeviceType (while checking arguments for " << op << ")";
    STD_TORCH_CHECK(false, oss.str());
  }
}

inline void check_same_gpu(const char* op,
                           const TensorArg& t1,
                           const TensorArg& t2) {
  const bool t1_cpu = t1.tensor.device().is_cpu();
  const bool t2_cpu = t2.tensor.device().is_cpu();
  if (t1_cpu || t2_cpu) {
    std::ostringstream oss;
    if (t1_cpu) {
      oss << "Tensor for " << t1 << " is on CPU, ";
    }
    if (t2_cpu) {
      oss << "Tensor for " << t2 << " is on CPU, ";
    }
    oss << "but expected " << ((t1_cpu && t2_cpu) ? "them" : "it")
        << " to be on GPU (while checking arguments for " << op << ')';
    STD_TORCH_CHECK(false, oss.str());
  }
  const auto d1 = t1.tensor.device();
  const auto d2 = t2.tensor.device();
  if (d1 == d2) {
    return;
  }
  std::ostringstream oss;
  oss << "Expected tensor for " << t1 << " to have the same device as tensor for "
      << t2 << "; but device ";
  append_device_string(oss, t1.tensor);
  oss << " does not equal ";
  append_device_string(oss, t2.tensor);
  oss << " (while checking arguments for " << op << ")";
  STD_TORCH_CHECK(false, oss.str());
}

inline void check_all_same_gpu(const char* op,
                               std::initializer_list<TensorArg> args) {
  const TensorArg* first = nullptr;
  for (const auto& arg : args) {
    if (first == nullptr) {
      first = &arg;
    } else {
      check_same_gpu(op, *first, arg);
    }
  }
}

}  // namespace
