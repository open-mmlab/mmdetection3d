// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
// Written by Shaoshuai Shi
// All Rights Reserved 2019.

#include <assert.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void roiaware_pool3d_launcher(int boxes_num, int pts_num, int channels,
                              int max_pts_each_voxel, int out_x, int out_y,
                              int out_z, const float *rois, const float *pts,
                              const float *pts_feature, int *argmax,
                              int *pts_idx_of_voxels, float *pooled_features,
                              int pool_method);

void roiaware_pool3d_backward_launcher(int boxes_num, int out_x, int out_y,
                                       int out_z, int channels,
                                       int max_pts_each_voxel,
                                       const int *pts_idx_of_voxels,
                                       const int *argmax, const float *grad_out,
                                       float *grad_in, int pool_method);

int roiaware_pool3d_gpu(at::Tensor rois, at::Tensor pts, at::Tensor pts_feature,
                        at::Tensor argmax, at::Tensor pts_idx_of_voxels,
                        at::Tensor pooled_features, int pool_method);

int roiaware_pool3d_gpu_backward(at::Tensor pts_idx_of_voxels,
                                 at::Tensor argmax, at::Tensor grad_out,
                                 at::Tensor grad_in, int pool_method);

int points_in_boxes_cpu(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                        at::Tensor pts_indices_tensor);

int points_in_boxes_part(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                         at::Tensor box_idx_of_points_tensor);

int points_in_boxes_all(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                        at::Tensor box_idx_of_points_tensor);

int roiaware_pool3d_gpu(at::Tensor rois, at::Tensor pts, at::Tensor pts_feature,
                        at::Tensor argmax, at::Tensor pts_idx_of_voxels,
                        at::Tensor pooled_features, int pool_method) {
  // params rois: (N, 7) [x, y, z, x_size, y_size, z_size, ry] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate
  // params pts_feature: (npoints, C)
  // params argmax: (N, out_x, out_y, out_z, C)
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params pooled_features: (N, out_x, out_y, out_z, C)
  // params pool_method: 0: max_pool 1: avg_pool

  CHECK_INPUT(rois);
  CHECK_INPUT(pts);
  CHECK_INPUT(pts_feature);
  CHECK_INPUT(argmax);
  CHECK_INPUT(pts_idx_of_voxels);
  CHECK_INPUT(pooled_features);

  int boxes_num = rois.size(0);
  int pts_num = pts.size(0);
  int channels = pts_feature.size(1);
  int max_pts_each_voxel = pts_idx_of_voxels.size(4);  // index 0 is the counter
  int out_x = pts_idx_of_voxels.size(1);
  int out_y = pts_idx_of_voxels.size(2);
  int out_z = pts_idx_of_voxels.size(3);
  assert((out_x < 256) && (out_y < 256) &&
         (out_z < 256));  // we encode index with 8bit

  const float *rois_data = rois.data_ptr<float>();
  const float *pts_data = pts.data_ptr<float>();
  const float *pts_feature_data = pts_feature.data_ptr<float>();
  int *argmax_data = argmax.data_ptr<int>();
  int *pts_idx_of_voxels_data = pts_idx_of_voxels.data_ptr<int>();
  float *pooled_features_data = pooled_features.data_ptr<float>();

  roiaware_pool3d_launcher(
      boxes_num, pts_num, channels, max_pts_each_voxel, out_x, out_y, out_z,
      rois_data, pts_data, pts_feature_data, argmax_data,
      pts_idx_of_voxels_data, pooled_features_data, pool_method);

  return 1;
}

int roiaware_pool3d_gpu_backward(at::Tensor pts_idx_of_voxels,
                                 at::Tensor argmax, at::Tensor grad_out,
                                 at::Tensor grad_in, int pool_method) {
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params argmax: (N, out_x, out_y, out_z, C)
  // params grad_out: (N, out_x, out_y, out_z, C)
  // params grad_in: (npoints, C), return value
  // params pool_method: 0: max_pool 1: avg_pool

  CHECK_INPUT(pts_idx_of_voxels);
  CHECK_INPUT(argmax);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(grad_in);

  int boxes_num = pts_idx_of_voxels.size(0);
  int out_x = pts_idx_of_voxels.size(1);
  int out_y = pts_idx_of_voxels.size(2);
  int out_z = pts_idx_of_voxels.size(3);
  int max_pts_each_voxel = pts_idx_of_voxels.size(4);  // index 0 is the counter
  int channels = grad_out.size(4);

  const int *pts_idx_of_voxels_data = pts_idx_of_voxels.data_ptr<int>();
  const int *argmax_data = argmax.data_ptr<int>();
  const float *grad_out_data = grad_out.data_ptr<float>();
  float *grad_in_data = grad_in.data_ptr<float>();

  roiaware_pool3d_backward_launcher(boxes_num, out_x, out_y, out_z, channels,
                                    max_pts_each_voxel, pts_idx_of_voxels_data,
                                    argmax_data, grad_out_data, grad_in_data,
                                    pool_method);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roiaware_pool3d_gpu, "roiaware pool3d forward (CUDA)");
  m.def("backward", &roiaware_pool3d_gpu_backward,
        "roiaware pool3d backward (CUDA)");
  m.def("points_in_boxes_part", &points_in_boxes_part,
        "points_in_boxes_part forward (CUDA)");
  m.def("points_in_boxes_all", &points_in_boxes_all,
        "points_in_boxes_all forward (CUDA)");
  m.def("points_in_boxes_cpu", &points_in_boxes_cpu,
        "points_in_boxes_cpu forward (CPU)");
}
