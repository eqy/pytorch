#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Types.h>

#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override")
#include <cudnn_frontend.h>
C10_DIAGNOSTIC_POP()

#include <ATen/native/utils/ParamsHash.h>
#include <ATen/native/cudnn/Pooling.h>


namespace at { namespace native {

template <typename T, typename KeyType>
struct JITCache {
std::mutex mutex;
std::unordered_map<KeyType, cudnn_frontend::ExecutionPlan, ParamsHash<KeyType>, ParamsEqual<KeyType>> engine_cache;

// TODO: is this thread safe if cache is updated? is pointer stale?
cudnn_frontend::ExecutionPlan* find(const KeyType& key) {
  std::lock_guard<std::mutex> guard(mutex);
  auto it = engine_cache.find(key);
  if (it == engine_cache.end()) {
    return nullptr;
  }
  // TODO: probably want ExecutionPlan copy constructor or better way to return
  return &(it->second);
}

void update(const KeyType& key, T& results) {
  std::lock_guard<std::mutex> guard(mutex);
  engine_cache.erase(key);
  engine_cache.emplace(key, std::move(results));
}

};


JITCache<cudnn_frontend::ExecutionPlan, PoolingParams> jit_cache;

void setPoolingParams(PoolingParams& params, Tensor& input, IntArrayRef kernel, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  memset(&params, 0, sizeof(params));
  params.device_id = at::cuda::current_device();
  params.dataType = getCudnnDataType(input);
  params.input_dim = (uint8_t) input.dim();
  TORCH_INTERNAL_ASSERT(params.input_dim == 4 || params.input_dim == 5, "cuDNN pooling only supports 4D or 5D input");
  for (uint8_t i = 0; params.input_dim; i++) {
    params.input_size[i] = (int) input.sizes()[i];
  }
  const uint8_t kernel_size = (uint8_t) kernel.size();
  const uint8_t stride_size = (uint8_t) stride.size();
  const uint8_t padding_size = (uint8_t) padding.size();
  const uint8_t dilation_size = (uint8_t) dilation.size();
  TORCH_INTERNAL_ASSERT(kernel_size <= POOLING_MAX_DIM && stride_size <= POOLING_MAX_DIM && dilation_size <= POOLING_MAX_DIM, "cuDNN pooling only supports 2D or 3D spatial dimensions");
  for (uint8_t i = 0; i < kernel_size; i++) {
    params.kernel_size[i] = kernel[i];
  }
  for (uint8_t i = 0; i < padding_size; i++) {
    params.padding[i] = padding[i];
  }
  for (uint8_t i = 0; i < dilation_size; i++) {
    params.dilation[i] = dilation[i];
  }
  params.memory_format = input.suggest_memory_format();
}

void cudnn_max_pooling_with_indices(const Tensor& input,  IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor& output, Tensor& indices) {
  int dims = dilation.size();
  for (int i = 0; i < dims; i++) {
    TORCH_INTERNAL_ASSERT(dilation[i] == 1, "cuDNN pooling does not currently support dilation != 1");
  }
}

}}
