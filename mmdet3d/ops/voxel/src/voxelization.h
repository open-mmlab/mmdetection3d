#pragma once
#include <torch/extension.h>

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

namespace voxelization {

int hard_voxelize_cpu(const at::Tensor &points, at::Tensor &voxels,
                      at::Tensor &coors, at::Tensor &num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim = 3);

void dynamic_voxelize_cpu(const at::Tensor &points, at::Tensor &coors,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const int NDim = 3);

std::vector<at::Tensor> dynamic_point_to_voxel_cpu(
    const at::Tensor &points, const at::Tensor &voxel_mapping,
    const std::vector<float> voxel_size, const std::vector<float> coors_range);

#ifdef WITH_CUDA
int hard_voxelize_gpu(const at::Tensor &points, at::Tensor &voxels,
                      at::Tensor &coors, at::Tensor &num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim = 3);

void dynamic_voxelize_gpu(const at::Tensor &points, at::Tensor &coors,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const int NDim = 3);

std::vector<torch::Tensor> dynamic_point_to_voxel_forward_gpu(const torch::Tensor &feats,
                                                              const torch::Tensor &coors,
                                                              const reduce_t reduce_type);

void dynamic_point_to_voxel_backward_gpu(torch::Tensor &grad_feats,
                                         const torch::Tensor &grad_reduced_feats,
                                         const torch::Tensor &feats,
                                         const torch::Tensor &reduced_feats,
                                         const torch::Tensor &coors_idx,
                                         const torch::Tensor &reduce_count,
                                         const reduce_t reduce_type);
#endif

// Interface for Python
inline int hard_voxelize(const at::Tensor &points, at::Tensor &voxels,
                         at::Tensor &coors, at::Tensor &num_points_per_voxel,
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

inline void dynamic_voxelize(const at::Tensor &points, at::Tensor &coors,
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

inline reduce_t convert_reduce_type(const std::string &reduce_type) {
  if (reduce_type == "max")
    return reduce_t::MAX;
  else if (reduce_type == "sum")
    return reduce_t::SUM;
  else if (reduce_type == "mean")
    return reduce_t::MEAN;
  else TORCH_CHECK(false, "do not support reduce type " + reduce_type)
  return reduce_t::SUM;
}

inline std::vector<torch::Tensor> dynamic_point_to_voxel_forward(const torch::Tensor &feats,
                                                                 const torch::Tensor &coors,
                                                                 const std::string &reduce_type) {
  if (feats.device().is_cuda()) {
#ifdef WITH_CUDA
    return dynamic_point_to_voxel_forward_gpu(feats, coors, convert_reduce_type(reduce_type));
#else
    TORCH_CHECK(false, "Not compiled with GPU support");
#endif
  }
  TORCH_CHECK(false, "do not support cpu yet");
  return std::vector<torch::Tensor>();
}

inline void dynamic_point_to_voxel_backward(torch::Tensor &grad_feats,
                                            const torch::Tensor &grad_reduced_feats,
                                            const torch::Tensor &feats,
                                            const torch::Tensor &reduced_feats,
                                            const torch::Tensor &coors_idx,
                                            const torch::Tensor &reduce_count,
                                            const std::string &reduce_type) {
  if (grad_feats.device().is_cuda()) {
#ifdef WITH_CUDA
    dynamic_point_to_voxel_backward_gpu(
        grad_feats, grad_reduced_feats, feats, reduced_feats, coors_idx, reduce_count,
        convert_reduce_type(reduce_type));
    return;
#else
    TORCH_CHECK(false, "Not compiled with GPU support");
#endif
  }
  TORCH_CHECK(false, "do not support cpu yet");
}

}  // namespace voxelization
