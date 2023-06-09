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
#include <aten/src/ATen/cudnn/Handle.h>

#include <c10/cuda/CUDACachingAllocator.h>

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

void setPoolingParams(PoolingParams& params, const Tensor& input, IntArrayRef kernel, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
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

uint8_t getAlignment(const Tensor &t) {
  // alignment are in bytes
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(t.data_ptr());
  for (; alignment < 32; alignment *= 2) {
    if (address % (alignment * 2)) {
      return alignment;
    }
  }
  return alignment;
}

bool
allowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}


void cudnn_max_pooling_with_indices(const Tensor& input,  IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor& output, Tensor& indices) {
  int dims = dilation.size();
  for (int i = 0; i < dims; i++) {
    TORCH_INTERNAL_ASSERT(dilation[i] == 1, "cuDNN pooling does not currently support dilation != 1");
  }
  PoolingParams params;
  setPoolingParams(params, input, kernel_size, stride, padding, dilation);
  cudnnHandle_t handle = getCudnnHandle();
  auto result = jit_cache.find(params);
  if (result) {

  } else { 
    auto xTensor = cudnn_frontend::TensorBuilder()
                     .setDim(input.sizes().size(), input.sizes().data())
                     .setStrides(input.strides().size(), input.strides().data())
                     .setId('x')
                     .setAlignment(getAlignment(input))
                     .setDataType(getCudnnDataType(input))
                     .build();
    auto yTensor = cudnn_frontend::TensorBuilder()
                     .setDim(output.sizes().size(), output.sizes().data())
                     .setStrides(output.strides().size(), output.strides().data())
                     .setId('y')
                     .setAlignment(getAlignment(output))
                     .setDataType(getCudnnDataType(output))
                     .build();
    auto idxTensor = cudnn_frontend::TensorBuilder()
                       .setDim(indices.sizes().size(), indices.sizes().data())
                       .setStrides(indices.strides().size(), indices.strides().data())
                       .setId('i') 
                       .setAlignment(getAlignment(indices))
                       .setDataType(getCudnnDataType(indices))
                       .build();

    const uint64_t spatial_dim = stride.size();
    int64_t postpadding[3] = {0, 0, 0};
     // Define the resample descriptor
    auto poolDesc = cudnn_frontend::ResampleDescBuilder_v8()
                        .setComputeType(CUDNN_DATA_FLOAT)
                        //.setNanPropagation(nanOpt)
                        .setResampleMode(CUDNN_RESAMPLE_MAXPOOL)
                        .setPaddingMode(CUDNN_NEG_INF_PAD)
                        .setSpatialDim(spatial_dim, kernel_size.data())
                        .setSpatialStride(spatial_dim, stride.data())
                        .setPrePadding(spatial_dim, padding.data())
                        .setPostPadding(3, postpadding)
                        .build(); 
    auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                       .setxDesc(xTensor)
                       .setyDesc(yTensor)
                       .setidxDesc(idxTensor)
                       .setResampleDesc(poolDesc)
                       .build();
    std::array<cudnn_frontend::Operation const*, 1> ops = {&pool_op};
    auto opGraph = cudnn_frontend::OperationGraphBuilder()
                   .setHandle(handle)
                   .setOperationGraph(ops.size(), ops.data())
                   .build();
    cudnn_frontend::EngineConfigList filtered_configs;
    auto statuses = cudnn_frontend::get_heuristics_list<2>({"heuristics_instant", "heuristics_fallback"}, opGraph, allowAll, filtered_configs, true);
    auto plan =
    cudnn_frontend::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
    auto workspace_size = plan.getWorkspaceSize();
    TORCH_WARN("workspace required: ", workspace_size);
    void* workspace_ptr = nullptr;
    if (workspace_size) {
      workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size).get();
    }
    void* data_ptrs[] = {input.data_ptr(), output.data_ptr(), indices.data_ptr()};
    int64_t uids[]    = {'x', 'y', 'i'};
    auto variantPack  = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(workspace_ptr)
                           .setDataPointers(3, data_ptrs)
                           .setUids(3, uids)
                           .build();
    cudnnStatus_t status = cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());
    TORCH_WARN("status", status);
  } 
}   
    
}}  
