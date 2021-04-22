#include <limits>

#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/core/Array.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/SortUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>

namespace {

template<typename scalar_t>
__global__ void sort_postprocess_kernel(const scalar_t *in, scalar_t *out, int64_t *index, const int2 *i_s_ptr, int nsegments, int nsort) {
  CUDA_KERNEL_LOOP(i, nsegments * nsort) {
    int segment = i / nsort;
    int j = i % nsort;

    int offset = segment * nsort;
    const scalar_t *in_ = in + offset;
    scalar_t *out_ = out + offset;
    int64_t *index_ = index + offset;
    const int2 *i_s_ptr_ = i_s_ptr + offset;

    int idx = i_s_ptr_[j].y;
    index_[j] = idx;
    out_[j] = in_[idx];
  }
}

}

namespace at { namespace native {

bool should_use_small_sort(const Tensor &self, int64_t dim) {
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t nsort = self.sizes()[dim];
  int64_t threshold;
  if (self.scalar_type() == kLong || self.scalar_type() == kDouble) {
    threshold = 1024;
  } else {
    threshold = 2048;
  }
  return nsort <= threshold;
}

std::vector<int64_t> infer_dense_strides_dim_last(const Tensor & self, int64_t dim);

void fillSliceWithIndex(Tensor& t,
                        int dim) {
  int64_t dims = t.dim();
  dims = dims == 0 ? 1 : dims;
  TORCH_CHECK(dims <= MAX_DIMS, "input tensor has too many dimensions");
  ptrdiff_t inElements = t.numel();
  if (inElements > 0) {
    int64_t sliceSize = t.dim() == 0 ? 1 : t.size(dim);
    ptrdiff_t numSlices = inElements / sliceSize;

    dim3 grid;
    if (!getGridFromTiles(numSlices, grid)) {
      AT_ERROR("Slice to fill with indices is too large");
    }

    int64_t maxThreads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    int64_t numThreads = sliceSize;
    if (numThreads > maxThreads) {
      numThreads = maxThreads;
    }

    dim3 block(numThreads);
#define FILL_INDEX(T, DIM)                                         \
    fillSliceWithIndex_kernel<T, DIM>                                     \
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(      \
        info, numSlices, sliceSize, info.strides[collapseDim]);    \
    C10_CUDA_KERNEL_LAUNCH_CHECK();


    if (at::cuda::detail::canUse32BitIndexMath(t)) {
      at::cuda::detail::TensorInfo<int64_t, unsigned int> info =
        at::cuda::detail::getTensorInfo<int64_t, uint32_t>(t);
      if (!t.dim()) {
        info.dims = 1;
        info.strides[0] = 1;
      }
      info.reduceDim(dim);
      int collapseDim = info.collapseDims(dim);
      if (info.isContiguous()) {
        FILL_INDEX(unsigned int, -2);
      } else {
        if (info.dims == 1) {
          FILL_INDEX(unsigned int, 1);
        } else if (info.dims == 2) {
          FILL_INDEX(unsigned int, 2);
        } else {
          FILL_INDEX(unsigned int, -1);
        }
      }
    } else {
      at::cuda::detail::TensorInfo<int64_t, uint64_t> info =
        at::cuda::detail::getTensorInfo<int64_t, uint64_t>(t);
      if (!t.dim()) {
        info.dims = 1;
        info.strides[0] = 1;
      }
      info.reduceDim(dim);
      int collapseDim = info.collapseDims(dim);

      // catch-all implementation
      FILL_INDEX(uint64_t, -1);
    }
#undef FILL_INDEX
  }
}

std::tuple<Tensor &,Tensor &> small_sort_out_cuda(const Tensor &self,
                                                  c10::optional<bool> stable,
                                                  Tensor &values,
                                                  Tensor &indices,
                                                  int dim, bool order) {
  TORCH_INTERNAL_ASSERT(stable.has_value(), "small_sort(): c10::optional<bool> for stable has to have value.");
  TORCH_CHECK(!stable.value(), "stable=True is not implemented on CUDA yet.");

  // from thc: sorted->values, indices->indices, input->self
  TensorArg self_arg{self, "self", 1}, values_arg{values, "values", 2}, indices_arg{indices, "indices", 3};
  checkAllSameGPU("small_sort", {self_arg, values_arg, indices_arg});

  if (!values.defined()) {
    values = at::empty_like(self, self.options());
  }
  if (!indices.defined()) {
    indices = at::empty_like(self, self.options().dtype(kLong));
  }

  // Make sure sufficient output space is allocated
  std::vector<int64_t> self_size = self.sizes().vec();
  at::native::resize_output(values, self_size);
  at::native::resize_output(indices, self_size);

  // How large are the slices that we are sorting?
  int64_t sliceSize = self.dim() == 0 ? 1 : self.size(dim);
  // FIXME: the k/v inplace sort along slice only works for size <=
  // 2048 at the moment
  // Workaround:
  int maxSliceSize;
  if (values.element_size() >= 8) {
    maxSliceSize = 1024;
  } else {
    maxSliceSize = 2048;
  }

  if (sliceSize <= maxSliceSize) {
    // Fill `indices` (the values) with the
    // slice-relative index.
    fillSliceWithIndex(indices, dim);

    // We sort k/v pairs in-place; copy unsorted input to output
    values.copy_(self);

    // Sort using our in-place k/v kernel that supports arbitrary
    // layout
    sortKeyValueInplace(values, indices, dim, order);
  } else {
    // This should be unreachable due to size threshold dispatching in cuda sort
    TORCH_INTERNAL_ASSERT(false);
  }

  return std::tuple<Tensor &, Tensor &>(values, indices);
}


// In alignment with default sort on a c++ map, this function
// will permute key and value tensors identically, and
// in such a way that the 'key' tensor is ordered numerically
void sortKeyValueInplace(Tensor& key,
                         Tensor& value,
                         int dim, bool dir) {
  TORCH_CHECK(key.sizes() == value.sizes(),
              "Key tensor must have same size as value tensor");
  int dims = value.dim();
  dims = dims == 0 ? 1 : dims;
  TORCH_CHECK(dims <= MAX_DIMS, "value tensor has too many dimensions");
  // if key and value tensors have the same size, we do not need to check both

  ptrdiff_t inElements = key.numel();

  if (inElements == 0) {
    return;
  }

  int64_t keySliceSize = key.dim() == 0 ? 1 : key.size(dim);
  ptrdiff_t keySlices = inElements / keySliceSize;

  // The amount of shared memory and block size is based on
  // 2^ceil(lg(n)); we choose that sorting implementation for a given
  // size.
  int64_t ceilPowerOf2 = nextHighestPowerOf2(keySliceSize);

  // FIXME: We'd have to find some other trick with Thrust to perform a
  // vectorized (key, value) sort by slice segment
  if (ceilPowerOf2 > 2048) {
    AT_ERROR("sortKeyValueInplace only works for sizes <= 2048 at present");
  }

  // The grid is based on the number of independent slices that we
  // have to sort; one block per slice
  dim3 grid;
  if (!getGridFromTiles(keySlices, grid)) {
    AT_ERROR("Slice to sort is too large");
  }

#define HANDLE_CASE(TYPE, A, SIZE)                                      \
  do {                                                                  \
    int blockSize = SIZE / 2;                                           \
    if (blockSize < 1) {                                                \
      blockSize = 1;                                                    \
    }                                                                   \
                                                                        \
    dim3 block(blockSize);                                              \
                                                                        \
    if (dir) {                                                          \
      bitonicSortKVInPlace<scalar_t, int64_t, A, -1,                    \
          GTComp<scalar_t, true>, TYPE, SIZE>                           \
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(        \
          keyInfo,                                                      \
          keySlices,                                                    \
          (TYPE) keySliceSize,                                          \
          (TYPE) keyInfo.strides[collapseKeyDim],                       \
          valueInfo,                                                    \
          (TYPE) valueInfo.strides[collapseValueDim],                   \
          GTComp<scalar_t, true>());                                    \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                   \
    } else {                                                            \
      bitonicSortKVInPlace<scalar_t, int64_t, A, -1,                    \
      LTComp<scalar_t, true>, TYPE, SIZE>                               \
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(        \
          keyInfo,                                                      \
          keySlices,                                                    \
          (TYPE) keySliceSize,                                          \
          (TYPE) keyInfo.strides[collapseKeyDim],                       \
          valueInfo,                                                    \
          (TYPE) valueInfo.strides[collapseValueDim],                   \
          LTComp<scalar_t, true>());                                    \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                   \
    }                                                                   \
  } while (0)

#define HANDLE_SORT_CASE(TYPE, A)                       \
  {                                                     \
    switch (ceilPowerOf2) {                             \
      case 2048:                                        \
      HANDLE_CASE(TYPE, A, 2048);                       \
      break;                                            \
      case 1024:                                        \
      case 512:                                         \
      case 256:                                         \
      HANDLE_CASE(TYPE, A, 1024);                       \
      break;                                            \
      case 128:                                         \
      case 64:                                          \
      HANDLE_CASE(TYPE, A, 128);                        \
      break;                                            \
      case 32:                                          \
      case 16:                                          \
      case 8:                                           \
      case 4:                                           \
      case 2:                                           \
      HANDLE_CASE(TYPE, A, 32);                         \
      break;                                            \
      case 1:                                           \
      /* Nothing to do, data already sorted */          \
      break;                                            \
      default:                                          \
      TORCH_INTERNAL_ASSERT(false);                                    \
    }                                                   \
  }

  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, key.scalar_type(), "sortKeyValueInplace", [&]  {
    if (at::cuda::detail::canUse32BitIndexMath(key)) {
      at::cuda::detail::TensorInfo<scalar_t, unsigned int> keyInfo =
        at::cuda::detail::getTensorInfo<scalar_t, unsigned int>(key);
      at::cuda::detail::TensorInfo<int64_t, unsigned int> valueInfo =
        at::cuda::detail::getTensorInfo<int64_t, unsigned int>(value);

      if (!key.dim()) {
        keyInfo.dims = 1;
        valueInfo.dims = 1;
        keyInfo.strides[0] = 1;
        valueInfo.strides[0] = 1;
      }
      keyInfo.reduceDim(dim);
      int collapseKeyDim = keyInfo.collapseDims(dim);
      valueInfo.reduceDim(dim);
      int collapseValueDim = valueInfo.collapseDims(dim);

      if (keyInfo.isContiguous()) {
        HANDLE_SORT_CASE(unsigned int, -2);
      } else {
        switch (keyInfo.dims) {
          case 2:
            HANDLE_SORT_CASE(unsigned int, 2);
            break;
          default:
            HANDLE_SORT_CASE(unsigned int, -1);
            break;
        }
      }

    } else {
      at::cuda::detail::TensorInfo<scalar_t, uint64_t> keyInfo =
        at::cuda::detail::getTensorInfo<scalar_t, uint64_t>(key);
      at::cuda::detail::TensorInfo<int64_t, uint64_t> valueInfo =
        at::cuda::detail::getTensorInfo<int64_t, uint64_t>(value);

      if (!key.dim()) {
        keyInfo.dims = 1;
        valueInfo.dims = 1;
        keyInfo.strides[0] = 1;
        valueInfo.strides[0] = 1;
      }
      keyInfo.reduceDim(dim);
      int collapseKeyDim = keyInfo.collapseDims(dim);
      valueInfo.reduceDim(dim);
      int collapseValueDim = valueInfo.collapseDims(dim);

      // int64_t case is rare, just instantiate the generic version
      HANDLE_SORT_CASE(uint64_t, -1);
    }
  });
#undef HANDLE_CASE
#undef HANDLE_SORT_CASE
#undef HANDLE_A_CASE
}

// We perform a vectorized segmented sort in cub.
// Say we are sorting a (2, 3) tensor. We have in flattened form:
// values       0.4 1.2 5.3 6.2 1.3 2.3
// indices        0   1   2   0   1   2
// segment_id     0   0   0   1   1   1

// First we sort by values, globally:
// values       6.2 5.3 2.3 1.2 1.3 0.4
// indices        0   2   2   1   1   0
// segment_id     1   0   1   0   1   0

// Then we stable sort by segment id:
// values       5.3 1.2 0.4 6.2 2.3 1.3
// indices        2   1   0   0   2   1
// segment_id     0   0   0   1   1   1

// This method can only work if the slice we are sorting (`dim`) is
// innermost, and both values and indices are contiguous. We do this
// by re-arranging the input into this form as needed, which will
// unfortunately allocate memory if the request is not in this form.
// Vectorized sort is slower than iterated sort if the number of
// slices is small (since we're sorting twice, instead of invoking a
// smaller sort `numSlices` times), but the cub sort
// implementation here is a catch-all, so we're not looking for
// efficiency, but instead correctness.
std::tuple<Tensor &,Tensor &> sort_out_stable_cuda(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
  // this algorithm is always stable
  TORCH_INTERNAL_ASSERT(stable.has_value(), "sort_out(): c10::optional<bool> for stable has to have value.");
  bool is_non_overlapping_and_dense = self.is_non_overlapping_and_dense();
  int64_t numel = self.numel();
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t nsort = self.sizes()[dim];

  TORCH_CHECK(nsort <= std::numeric_limits<int>::max(),
    "The dimension being sorted can not have more than INT_MAX elsments.");

  // use inplace algorithm for smaller input sizes
  if (should_use_small_sort(self, dim)) {
    return small_sort_out_cuda(self, stable, values, indices, dim, descending);
  }

  if (ndim == 0) {
    if (!values.defined()) {
      values = self.clone();
    } else {
      values.resize_as_(self);
      values.copy_(self);
    }
    if (!indices.defined()) {
      indices = at::zeros({}, self.options().dtype(kLong));
    } else {
      indices.resize_as_(self);
      indices.zero_();
    }
    return std::forward_as_tuple(values, indices);
  }

  Tensor self_;
  if (is_non_overlapping_and_dense && self.stride(dim) == 1) {
    self_ = self;
  } else {
    auto new_strides_unsort = infer_dense_strides_dim_last(self, dim);
    self_ = at::empty_strided(self.sizes(), new_strides_unsort, self.options());
    self_.copy_(self);
  }

  Tensor values_tmp, indices_tmp;
  void *values_ptr_;
  int64_t *indices_ptr;
  if (!values.defined()) {
    if (is_non_overlapping_and_dense) {
      values = at::empty_strided(self.sizes(), self.strides(), self.options());
    } else {
      auto strides = at::infer_dense_strides(self.sizes(), self.strides());
      values = at::empty_strided(self.sizes(), strides, self.options());
    }
  } else {
    TORCH_CHECK(self_.scalar_type() == values.scalar_type(),
      "Unexpected dtype for values, expect ", self_.scalar_type(), ", got ", values.scalar_type());
    values.resize_as_(self);
  }
  if (values.strides() != self_.strides()) {
    values_tmp = at::empty_strided(self_.sizes(), self_.strides(), self_.options());
    values_ptr_ = values_tmp.data_ptr();
  } else {
    values_ptr_ = values.data_ptr();
  }

  if (!indices.defined()) {
    if (is_non_overlapping_and_dense) {
      indices = at::empty_strided(self.sizes(), self.strides(), self.options().dtype(kLong));
    } else {
      auto strides = at::infer_dense_strides(self.sizes(), self.strides());
      indices = at::empty_strided(self.sizes(), strides, self.options().dtype(kLong));
    }
  } else {
    TORCH_CHECK(kLong == indices.scalar_type(),
      "Unexpected dtype for values, expect torch.long, got ", indices.scalar_type());
    indices.resize_as_(self);
  }
  if (indices.strides() != self_.strides()) {
    indices_tmp = at::empty_strided(self_.sizes(), self_.strides(), self_.options().dtype(kLong));
    indices_ptr = indices_tmp.data_ptr<int64_t>();
  } else {
    indices_ptr = indices.data_ptr<int64_t>();
  }

  if (numel == 0) {
    return std::forward_as_tuple(values, indices);
  }

  int64_t numel_or_intmax = std::min(numel, static_cast<int64_t>(std::numeric_limits<int>::max()));
  int64_t nbatch = (numel_or_intmax / nsort) * nsort;

  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, self_.scalar_type(), "sort", [&]{
    const scalar_t *self_ptr = self_.data_ptr<scalar_t>();
    auto values_ptr = reinterpret_cast<scalar_t *>(values_ptr_);
    int64_t remaining = numel;
    while (remaining > 0) {
      int64_t n = std::min(remaining, nbatch);
      int64_t nsegments = n / nsort;
      int64_t segment_bits = std::max<int64_t>(1L, static_cast<int64_t>(std::ceil(std::log2(nsegments))));

      auto int_options = indices.options().dtype(kInt);
      auto indices_and_segment = at::empty({nsegments, nsort, 2}, int_options);
      indices_and_segment.select(-1, 0).copy_(  // segment id
        at::arange(nsegments, int_options).view({nsegments, 1}).expand({nsegments, nsort}));
      indices_and_segment.select(-1, 1).copy_(  // reverse indices
        at::arange(nsort, int_options).view({1, nsort}).expand({nsegments, nsort}));

      auto i_s_ptr = reinterpret_cast<int2 *>(indices_and_segment.data_ptr<int>());
      auto indices_and_segment2 = at::empty_like(indices_and_segment);
      auto i_s_ptr2 = reinterpret_cast<int2 *>(indices_and_segment2.data_ptr<int>());

      at::cuda::cub::sort_pairs<scalar_t, int2>(
        self_ptr, nullptr, i_s_ptr, i_s_ptr2,
        n, descending);

      TORCH_INTERNAL_ASSERT(segment_bits <= 32);

      // sort on lower 32bits, i.e. segment index
      at::cuda::cub::sort_keys<int64_t>(
        reinterpret_cast<int64_t *>(i_s_ptr2), reinterpret_cast<int64_t *>(i_s_ptr),
        n, false, 0, segment_bits);

      sort_postprocess_kernel<<<(n + 511) / 512, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
        self_ptr, values_ptr, indices_ptr, i_s_ptr, nsegments, nsort);

      remaining -= n;
      self_ptr += n;
      values_ptr += n;
      indices_ptr += n;
    }
  });

  if (values_tmp.defined()) {
    values.copy_(values_tmp);
  }
  if (indices_tmp.defined()) {
    indices.copy_(indices_tmp);
  }
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor &,Tensor &> sort_out_cuda(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
  return sort_out_stable_cuda(self, /*stable=*/false, dim, descending, values, indices);
}

std::tuple<Tensor,Tensor> sort_stable_cuda(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
  Tensor values, indices;
  return sort_out_stable_cuda(self, stable, dim, descending, values, indices);
}

std::tuple<Tensor,Tensor> sort_cuda(const Tensor & self, int64_t dim, bool descending) {  int64_t threshold;
  return sort_stable_cuda(self, /*stable=*/false, dim, descending);
}

}}  // namespace at::native
