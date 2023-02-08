#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace {
int const threadsPerBlock = 512;
int const maxGridDim = 50000;
}  // namespace

__device__ __forceinline__ static void reduceMax(float *address, float val) {
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old || __int_as_float(old) < val);
}

__device__ __forceinline__ static void reduceMax(double *address, double val) {
  unsigned long long *address_as_ull =
      reinterpret_cast<unsigned long long *>(address);
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed,
        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old || __longlong_as_double(old) < val);
}

// get rid of meaningless warnings when compiling host code
#ifdef __CUDA_ARCH__
__device__ __forceinline__ static void reduceAdd(float *address, float val) {
#if (__CUDA_ARCH__ < 200)
#warning \
    "compute capability lower than 2.x. fall back to use CAS version of atomicAdd for float32"
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(val + __int_as_float(assumed)));
  } while (assumed != old);
#else
  atomicAdd(address, val);
#endif
}

__device__ __forceinline__ static void reduceAdd(double *address, double val) {
#if (__CUDA_ARCH__ < 600)
#warning \
    "compute capability lower than 6.x. fall back to use CAS version of atomicAdd for float64"
  unsigned long long *address_as_ull =
      reinterpret_cast<unsigned long long *>(address);
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
#else
  atomicAdd(address, val);
#endif
}
#endif

template <typename T>
__global__ void
feats_reduce_kernel(const T *feats, const int32_t *coors_map,
                    T *reduced_feats, // shall be 0 at initialization
                    const int num_input, const int num_feats,
                    const reduce_t reduce_type) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_input;
       x += gridDim.x * blockDim.x) {
    int32_t reduce_to = coors_map[x];
    if (reduce_to == -1) continue;

    const T *feats_offset = feats + x * num_feats;
    T *reduced_feats_offset = reduced_feats + reduce_to * num_feats;
    if (reduce_type == reduce_t::MAX) {
      for (int i = 0; i < num_feats; i++) {
        reduceMax(&reduced_feats_offset[i], feats_offset[i]);
      }
    } else {
      for (int i = 0; i < num_feats; i++) {
        reduceAdd(&reduced_feats_offset[i], feats_offset[i]);
      }
    }
  }
}

template <typename T>
__global__ void add_reduce_traceback_grad_kernel(
    T *grad_feats, const T *grad_reduced_feats, const int32_t *coors_map,
    const int32_t *reduce_count, const int num_input, const int num_feats,
    const reduce_t reduce_type) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_input;
       x += gridDim.x * blockDim.x) {
    int32_t reduce_to = coors_map[x];
    if (reduce_to == -1) {
      continue;
    }

    const int input_offset = x * num_feats;
    T *grad_feats_offset = grad_feats + input_offset;
    const int reduced_offset = reduce_to * num_feats;
    const T *grad_reduced_feats_offset = grad_reduced_feats + reduced_offset;

    if (reduce_type == reduce_t::SUM) {
      for (int i = 0; i < num_feats; i++) {
        grad_feats_offset[i] = grad_reduced_feats_offset[i];
      }
    } else if (reduce_type == reduce_t::MEAN) {
      for (int i = 0; i < num_feats; i++) {
        grad_feats_offset[i] = grad_reduced_feats_offset[i] /
                               static_cast<T>(reduce_count[reduce_to]);
      }
    }
  }
}

template <typename T>
__global__ void max_reduce_traceback_scatter_idx_kernel(
    const T *feats, const T *reduced_feats, int32_t *reduce_from,
    const int32_t *coors_map, const int num_input, const int num_feats) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_input;
       x += gridDim.x * blockDim.x) {
    int32_t reduce_to = coors_map[x];

    const int input_offset = x * num_feats;
    const T *feats_offset = feats + input_offset;

    if (reduce_to == -1) {
      continue;
    }

    const int reduced_offset = reduce_to * num_feats;
    const T *reduced_feats_offset = reduced_feats + reduced_offset;
    int32_t *reduce_from_offset = reduce_from + reduced_offset;

    for (int i = 0; i < num_feats; i++) {
      if (feats_offset[i] == reduced_feats_offset[i]) {
        atomicMin(&reduce_from_offset[i], static_cast<int32_t>(x));
      }
    }
  }
}

template <typename T>
__global__ void max_reduce_scatter_grad_kernel(T *grad_feats,
                                               const T *grad_reduced_feats,
                                               const int32_t *reduce_from,
                                               const int num_reduced,
                                               const int num_feats) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_reduced;
       x += gridDim.x * blockDim.x) {
    const int reduced_offset = x * num_feats;
    const int32_t *scatter_to_offset = reduce_from + reduced_offset;
    const T *grad_reduced_feats_offset = grad_reduced_feats + reduced_offset;

    for (int i = 0; i < num_feats; i++) {
      grad_feats[scatter_to_offset[i] * num_feats + i] =
          grad_reduced_feats_offset[i];
    }
  }
}

namespace voxelization {

std::vector<at::Tensor> dynamic_point_to_voxel_forward_gpu(
    const at::Tensor &feats, const at::Tensor &coors,
    const reduce_t reduce_type) {
  CHECK_INPUT(feats);
  CHECK_INPUT(coors);

  const int num_input = feats.size(0);
  const int num_feats = feats.size(1);

  if (num_input == 0)
    return {feats.clone().detach(),
            coors.clone().detach(),
            coors.new_empty({0}, torch::kInt32),
            coors.new_empty({0}, torch::kInt32)};

  at::Tensor out_coors;
  at::Tensor coors_map;
  at::Tensor reduce_count;

  auto coors_clean = coors.masked_fill(coors.lt(0).any(-1, true), -1);

  std::tie(out_coors, coors_map, reduce_count) =
      at::unique_dim(coors_clean, 0, true, true, true);

  if (out_coors.index({0, 0}).lt(0).item<bool>()) {
    // the first element of out_coors (-1,-1,-1) and should be removed
    out_coors = out_coors.slice(0, 1);
    reduce_count = reduce_count.slice(0, 1);
    coors_map = coors_map - 1;
  }

  coors_map = coors_map.to(torch::kInt32);
  reduce_count = reduce_count.to(torch::kInt32);

  auto reduced_feats =
      at::empty({out_coors.size(0), num_feats}, feats.options());

  AT_DISPATCH_FLOATING_TYPES(
      feats.scalar_type(), "feats_reduce_kernel", ([&] {
    if (reduce_type == reduce_t::MAX)
      reduced_feats.fill_(-std::numeric_limits<scalar_t>::infinity());
    else
      reduced_feats.fill_(static_cast<scalar_t>(0));

    dim3 blocks(std::min(at::cuda::ATenCeilDiv(num_input, threadsPerBlock),
                         maxGridDim));
    dim3 threads(threadsPerBlock);
    feats_reduce_kernel<<<blocks, threads>>>(
        feats.data_ptr<scalar_t>(), coors_map.data_ptr<int32_t>(),
        reduced_feats.data_ptr<scalar_t>(), num_input, num_feats, reduce_type);
    if (reduce_type == reduce_t::MEAN)
      reduced_feats /= reduce_count.unsqueeze(-1).to(reduced_feats.dtype());
  }));
  AT_CUDA_CHECK(cudaGetLastError());

  return {reduced_feats, out_coors, coors_map, reduce_count};
}

void dynamic_point_to_voxel_backward_gpu(at::Tensor &grad_feats,
                                         const at::Tensor &grad_reduced_feats,
                                         const at::Tensor &feats,
                                         const at::Tensor &reduced_feats,
                                         const at::Tensor &coors_map,
                                         const at::Tensor &reduce_count,
                                         const reduce_t reduce_type) {
  CHECK_INPUT(grad_feats);
  CHECK_INPUT(grad_reduced_feats);
  CHECK_INPUT(feats);
  CHECK_INPUT(reduced_feats);
  CHECK_INPUT(coors_map);
  CHECK_INPUT(reduce_count);

  const int num_input = feats.size(0);
  const int num_reduced = reduced_feats.size(0);
  const int num_feats = feats.size(1);

  grad_feats.fill_(0);
  // copy voxel grad to points

  if (num_input == 0 || num_reduced == 0) return;

  if (reduce_type == reduce_t::MEAN || reduce_type == reduce_t::SUM) {
    AT_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.scalar_type(), "add_reduce_traceback_grad_kernel",
        ([&] {
          dim3 blocks(std::min(
              at::cuda::ATenCeilDiv(num_input, threadsPerBlock), maxGridDim));
          dim3 threads(threadsPerBlock);
          add_reduce_traceback_grad_kernel<<<blocks, threads>>>(
              grad_feats.data_ptr<scalar_t>(),
              grad_reduced_feats.data_ptr<scalar_t>(),
              coors_map.data_ptr<int32_t>(), reduce_count.data_ptr<int32_t>(),
              num_input, num_feats, reduce_type);
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  } else {
    auto reduce_from = at::full({num_reduced, num_feats}, num_input,
                                coors_map.options().dtype(torch::kInt32));
    AT_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.scalar_type(),
        "max_reduce_traceback_scatter_idx_kernel", ([&] {
          dim3 blocks(std::min(
              at::cuda::ATenCeilDiv(num_input, threadsPerBlock), maxGridDim));
          dim3 threads(threadsPerBlock);
          max_reduce_traceback_scatter_idx_kernel<<<blocks, threads>>>(
              feats.data_ptr<scalar_t>(), reduced_feats.data_ptr<scalar_t>(),
              reduce_from.data_ptr<int32_t>(), coors_map.data_ptr<int32_t>(),
              num_input, num_feats);
        }));
    AT_CUDA_CHECK(cudaGetLastError());

    AT_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.scalar_type(),
        "max_reduce_traceback_scatter_idx_kernel", ([&] {
          dim3 blocks(std::min(
              at::cuda::ATenCeilDiv(num_reduced, threadsPerBlock), maxGridDim));
          dim3 threads(threadsPerBlock);
          max_reduce_scatter_grad_kernel<<<blocks, threads>>>(
              grad_feats.data_ptr<scalar_t>(),
              grad_reduced_feats.data_ptr<scalar_t>(),
              reduce_from.data_ptr<int32_t>(), num_reduced, num_feats);
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  }
  return;
}

}  // namespace voxelization
