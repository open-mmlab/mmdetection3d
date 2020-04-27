#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;
}

template <typename T, typename T_int>
__global__ void scatter_point_to_voxel_kernel(
    const T* points, T_int* coor, T_int* point_to_voxelidx,
    T_int* coor_to_voxelidx, T* voxels, T_int* coors, const int num_features,
    const int num_points, const int max_points, const int NDim) {
  const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  if (index >= num_points) return;

  int num = point_to_voxelidx[index];
  int voxelidx = coor_to_voxelidx[index];
  if (num > -1 && voxelidx > -1) {
    const int feature_per_thread = num_features / 4;

    int start = threadIdx.y * feature_per_thread;
    auto voxels_offset =
        voxels + voxelidx * max_points * num_features + num * num_features;
    auto points_offset = points + index * num_features;
    for (int k = start; k < start + feature_per_thread; k++) {
      voxels_offset[k] = points_offset[k];
    }
    if (num == 0 && start < NDim) {
      auto coors_offset = coors + voxelidx * NDim;
      auto coor_offset = coor + index * NDim;
      for (int k = start; k < NDim; k++) {
        coors_offset[k] = coor_offset[k];
      }
    }
  }
}

template <typename T, typename T_int>
__global__ void map_voxel_to_point_kernel(
    T* points, T* voxels, T_int* point_to_voxelidx, T_int* coor_to_voxelidx,
    const int num_features, const int num_points, const int max_points) {
  const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  if (index >= num_points) return;
  auto num = point_to_voxelidx[index];
  if (num > -1) {
    const int feature_per_thread = num_features / 4;
    auto voxelidx = coor_to_voxelidx[index];

    int start = threadIdx.y * feature_per_thread;
    auto voxels_offset =
        voxels + voxelidx * max_points * num_features + num * num_features;
    auto points_offset = points + index * num_features;
    for (int k = start; k < start + feature_per_thread; k++) {
      points_offset[k] = voxels_offset[k];
    }
  }
}

template <typename T_int>
__global__ void point_to_voxelidx_kernel(const T_int* coor,
                                         T_int* point_to_voxelidx,
                                         T_int* point_to_pointidx,
                                         const int num_points, const int NDim) {
  const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  auto coor_offset = coor + index * NDim;
  // skip invalid points
  if ((index >= num_points) || (coor_offset[0] == -1)) return;

  int num = 0;
  int coor_x = coor_offset[0];
  int coor_y = coor_offset[1];
  int coor_z = coor_offset[2];
  // only calculate the coors before this coor[index]
  for (int i = 0; i < index; ++i) {
    auto prev_coor = coor + i * NDim;
    if (prev_coor[0] == -1) continue;

    // record voxel
    if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
        (prev_coor[2] == coor_z)) {
      num++;
      if (num == 1) {
        point_to_pointidx[index] = i;
      }
    }
  }
  if (num == 0) {
    point_to_pointidx[index] = index;
  }
  point_to_voxelidx[index] = num;
}

template <typename T_int>
__global__ void determin_voxel_num(
    const T_int* coor, T_int* num_points_per_voxel, T_int* point_to_voxelidx,
    T_int* point_to_pointidx, T_int* coor_to_voxelidx, T_int* voxel_num,
    T_int* max_points, const int num_points, const int NDim) {
  // only calculate the coors before this coor[index]
  for (int i = 0; i < num_points; ++i) {
    auto coor_offset = coor + i * NDim;
    if (coor_offset[0] == -1) continue;
    int point_pos_in_voxel = point_to_voxelidx[i];
    // record voxel
    if (point_pos_in_voxel == -1) {
      // out of max_points or invalid point
      printf("point_pos_in_voxel == -1, point:%d", i);
      continue;
    } else if (point_pos_in_voxel == 0) {
      // record new voxel
      int voxelidx = voxel_num[0];
      voxel_num[0] += 1;
      coor_to_voxelidx[i] = voxelidx;
      num_points_per_voxel[voxelidx] = 1;
    } else {
      int point_idx = point_to_pointidx[i];
      int voxelidx = coor_to_voxelidx[point_idx];
      if (voxelidx != -1) {
        num_points_per_voxel[voxelidx] += 1;
        coor_to_voxelidx[i] = voxelidx;
        max_points[0] = max(max_points[0], point_pos_in_voxel + 1);
      } else {
        printf("voxelidx = -1, point:%d", i);
      }
    }
  }
}

namespace voxelization {

std::vector<at::Tensor> dynamic_point_to_voxel_forward_gpu(
    const at::Tensor& points, const at::Tensor& voxel_mapping,
    const std::vector<float> voxel_size, const std::vector<float> coors_range) {
  CHECK_INPUT(points);
  at::cuda::CUDAGuard device_guard(points.device());

  const int NDim = voxel_mapping.size(1);
  const int num_points = points.size(0);
  const int num_features = points.size(1);

  std::vector<int> grid_size(NDim);
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }

  // assume the mapping is already given
  auto point_to_pointidx = -at::ones(
      {
          num_points,
      },
      voxel_mapping.options());
  auto point_to_voxelidx = -at::ones(
      {
          num_points,
      },
      voxel_mapping.options());
  auto max_points = at::zeros(
      {
          1,
      },
      voxel_mapping.options());  // must be zero from the begining

  int col_blocks = at::cuda::ATenCeilDiv(num_points, threadsPerBlock);
  dim3 blocks(col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t map_stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(
      voxel_mapping.scalar_type(), "determin_duplicate", ([&] {
        point_to_voxelidx_kernel<int><<<blocks, threads, 0, map_stream>>>(
            voxel_mapping.data_ptr<int>(), point_to_voxelidx.data_ptr<int>(),
            point_to_pointidx.data_ptr<int>(), num_points, NDim);
      }));
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  // make the logic in the CUDA device could accelerate about 10 times
  auto num_points_per_voxel = at::zeros(
      {
          num_points,
      },
      voxel_mapping.options());
  auto coor_to_voxelidx = -at::ones(
      {
          num_points,
      },
      voxel_mapping.options());
  auto voxel_num = at::zeros(
      {
          1,
      },
      voxel_mapping.options());  // must be zero from the begining
  cudaStream_t logic_stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(
      voxel_mapping.scalar_type(), "determin_duplicate", ([&] {
        determin_voxel_num<int><<<1, 1, 0, logic_stream>>>(
            voxel_mapping.data_ptr<int>(), num_points_per_voxel.data_ptr<int>(),
            point_to_voxelidx.data_ptr<int>(),
            point_to_pointidx.data_ptr<int>(), coor_to_voxelidx.data_ptr<int>(),
            voxel_num.data_ptr<int>(), max_points.data_ptr<int>(), num_points,
            NDim);
      }));
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  // some temporary data
  auto max_points_cpu = max_points.to(at::kCPU);
  int max_points_int = max_points_cpu.data_ptr<int>()[0];
  auto voxel_num_cpu = voxel_num.to(at::kCPU);
  int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];
  at::Tensor coors =
      at::zeros({voxel_num_int, NDim}, points.options().dtype(at::kInt));
  at::Tensor voxels = at::zeros({voxel_num_int, max_points_int, num_features},
                                points.options());

  // copy point features to voxels
  dim3 cp_threads(threadsPerBlock, 4);
  cudaStream_t cp_stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "scatter_point_to_voxel", ([&] {
        scatter_point_to_voxel_kernel<float, int>
            <<<blocks, cp_threads, 0, cp_stream>>>(
                points.data_ptr<float>(), voxel_mapping.data_ptr<int>(),
                point_to_voxelidx.data_ptr<int>(),
                coor_to_voxelidx.data_ptr<int>(), voxels.data_ptr<float>(),
                coors.data_ptr<int>(), num_features, num_points, max_points_int,
                NDim);
      }));
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  at::Tensor num_points_per_voxel_out =
      num_points_per_voxel.slice(/*dim=*/0, /*start=*/0, /*end=*/voxel_num_int);
  return {voxels, coors, num_points_per_voxel_out, point_to_voxelidx,
          coor_to_voxelidx};
}

void dynamic_point_to_voxel_backward_gpu(at::Tensor& grad_input_points,
                                         const at::Tensor& grad_output_voxels,
                                         const at::Tensor& point_to_voxelidx,
                                         const at::Tensor& coor_to_voxelidx) {
  CHECK_INPUT(grad_input_points);
  CHECK_INPUT(grad_output_voxels);
  CHECK_INPUT(point_to_voxelidx);
  CHECK_INPUT(coor_to_voxelidx);
  at::cuda::CUDAGuard device_guard(grad_input_points.device());

  const int num_points = grad_input_points.size(0);
  const int num_features = grad_input_points.size(1);
  const int max_points = grad_output_voxels.size(1);

  // copy voxel grad to points
  int col_blocks = at::cuda::ATenCeilDiv(num_points, threadsPerBlock);
  dim3 blocks(col_blocks);
  dim3 cp_threads(threadsPerBlock, 4);
  cudaStream_t cp_stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(grad_input_points.scalar_type(),
                        "scatter_point_to_voxel", ([&] {
                          map_voxel_to_point_kernel<float, int>
                              <<<blocks, cp_threads, 0, cp_stream>>>(
                                  grad_input_points.data_ptr<float>(),
                                  grad_output_voxels.data_ptr<float>(),
                                  point_to_voxelidx.data_ptr<int>(),
                                  coor_to_voxelidx.data_ptr<int>(),
                                  num_features, num_points, max_points);
                        }));
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  return;
}

}  // namespace voxelization
