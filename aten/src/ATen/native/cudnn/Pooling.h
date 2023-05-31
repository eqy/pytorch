#pragma once
#include <ATen/core/Tensor.h>

namespace at { namespace native {

#define POOLING_MAX_DIM 3

struct PoolingParams {
  c10::DeviceIndex device_id;
  cudnnDataType_t dataType;
  int input_size[POOLING_MAX_DIM + 2];
  uint8_t input_dim;
  at::MemoryFormat memory_format;
  int kernel_size[POOLING_MAX_DIM];
  int padding[POOLING_MAX_DIM];
  int stride[POOLING_MAX_DIM];
  int dilation[POOLING_MAX_DIM];
};

static bool use_cudnn_v8_jit_pooling() {
  static bool flag = c10::utils::check_env("TORCH_CUDNN_JIT_POOLING_ENABLED") == true;
  return flag;
}

void setPoolingParams(PoolingParams& params, Tensor& input, IntArrayRef kernel, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation);

std::tuple<Tensor, Tensor> cudnn_pooling_with_indices(
  const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation);

}}
