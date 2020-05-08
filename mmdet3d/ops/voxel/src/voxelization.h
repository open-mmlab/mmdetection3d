#pragma once
#include <torch/extension.h>

namespace voxelization {

int hard_voxelize_cpu(const at::Tensor& points, at::Tensor& voxels,
                      at::Tensor& coors, at::Tensor& num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim = 3);

void dynamic_voxelize_cpu(const at::Tensor& points, at::Tensor& coors,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const int NDim = 3);

std::vector<at::Tensor> dynamic_point_to_voxel_cpu(
    const at::Tensor& points, const at::Tensor& voxel_mapping,
    const std::vector<float> voxel_size, const std::vector<float> coors_range);

#ifdef WITH_CUDA
int hard_voxelize_gpu(const at::Tensor& points, at::Tensor& voxels,
                      at::Tensor& coors, at::Tensor& num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim = 3);

void dynamic_voxelize_gpu(const at::Tensor& points, at::Tensor& coors,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const int NDim = 3);

std::vector<at::Tensor> dynamic_point_to_voxel_forward_gpu(
    const at::Tensor& points, const at::Tensor& voxel_mapping,
    const std::vector<float> voxel_size, const std::vector<float> coors_range);

void dynamic_point_to_voxel_backward_gpu(at::Tensor& grad_input_points,
                                         const at::Tensor& grad_output_voxels,
                                         const at::Tensor& point_to_voxelidx,
                                         const at::Tensor& coor_to_voxelidx);
#endif

// Interface for Python
inline int hard_voxelize(const at::Tensor& points, at::Tensor& voxels,
                         at::Tensor& coors, at::Tensor& num_points_per_voxel,
                         const std::vector<float> voxel_size,
                         const std::vector<float> coors_range,
                         const int max_points, const int max_voxels,
                         const int NDim = 3) {
  if (points.device().is_cuda()) {
#ifdef WITH_CUDA
    return hard_voxelize_gpu(points, voxels, coors, num_points_per_voxel,
                             voxel_size, coors_range, max_points, max_voxels,
                             NDim);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return hard_voxelize_cpu(points, voxels, coors, num_points_per_voxel,
                           voxel_size, coors_range, max_points, max_voxels,
                           NDim);
}

inline void dynamic_voxelize(const at::Tensor& points, at::Tensor& coors,
                             const std::vector<float> voxel_size,
                             const std::vector<float> coors_range,
                             const int NDim = 3) {
  if (points.device().is_cuda()) {
#ifdef WITH_CUDA
    return dynamic_voxelize_gpu(points, coors, voxel_size, coors_range, NDim);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return dynamic_voxelize_cpu(points, coors, voxel_size, coors_range, NDim);
}

inline std::vector<torch::Tensor> dynamic_point_to_voxel_forward(
    const at::Tensor& points, const at::Tensor& voxel_mapping,
    const std::vector<float> voxel_size, const std::vector<float> coors_range) {
  if (points.device().is_cuda()) {
#ifdef WITH_CUDA
    return dynamic_point_to_voxel_forward_gpu(points, voxel_mapping, voxel_size,
                                              coors_range);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return dynamic_point_to_voxel_cpu(points, voxel_mapping, voxel_size,
                                    coors_range);
}

inline void dynamic_point_to_voxel_backward(
    at::Tensor& grad_input_points, const at::Tensor& grad_output_voxels,
    const at::Tensor& point_to_voxelidx, const at::Tensor& coor_to_voxelidx) {
  if (grad_input_points.device().is_cuda()) {
#ifdef WITH_CUDA
    return dynamic_point_to_voxel_backward_gpu(
        grad_input_points, grad_output_voxels, point_to_voxelidx,
        coor_to_voxelidx);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  // return dynamic_point_to_voxel_cpu(points,
  //                                  voxel_mapping,
  //                                  voxel_size,
  //                                  coors_range);
}

}  // namespace voxelization
