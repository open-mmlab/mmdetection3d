#include <ATen/TensorUtils.h>
#include <torch/extension.h>
// #include "voxelization.h"

namespace {

template <typename T_int>
void determin_max_points_kernel(
    torch::TensorAccessor<T_int, 2> coor,
    torch::TensorAccessor<T_int, 1> point_to_voxelidx,
    torch::TensorAccessor<T_int, 1> num_points_per_voxel,
    torch::TensorAccessor<T_int, 3> coor_to_voxelidx, int& voxel_num,
    int& max_points, const int num_points) {
  int voxelidx, num;
  for (int i = 0; i < num_points; ++i) {
    if (coor[i][0] == -1) continue;
    voxelidx = coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]];

    // record voxel
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      voxel_num += 1;
      coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]] = voxelidx;
    }

    // put points into voxel
    num = num_points_per_voxel[voxelidx];
    point_to_voxelidx[i] = num;
    num_points_per_voxel[voxelidx] += 1;

    // update max points per voxel
    max_points = std::max(max_points, num + 1);
  }

  return;
}

template <typename T, typename T_int>
void scatter_point_to_voxel_kernel(
    const torch::TensorAccessor<T, 2> points,
    torch::TensorAccessor<T_int, 2> coor,
    torch::TensorAccessor<T_int, 1> point_to_voxelidx,
    torch::TensorAccessor<T_int, 3> coor_to_voxelidx,
    torch::TensorAccessor<T, 3> voxels,
    torch::TensorAccessor<T_int, 2> voxel_coors, const int num_features,
    const int num_points, const int NDim) {
  for (int i = 0; i < num_points; ++i) {
    int num = point_to_voxelidx[i];
    int voxelidx = coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]];
    for (int k = 0; k < num_features; ++k) {
      voxels[voxelidx][num][k] = points[i][k];
    }
    for (int k = 0; k < NDim; ++k) {
      voxel_coors[voxelidx][k] = coor[i][k];
    }
  }
}

}  // namespace

namespace voxelization {

std::vector<at::Tensor> dynamic_point_to_voxel_cpu(
    const at::Tensor& points, const at::Tensor& voxel_mapping,
    const std::vector<float> voxel_size, const std::vector<float> coors_range) {
  // current version tooks about 0.02s_0.03s for one frame on cpu
  // check device
  AT_ASSERTM(points.device().is_cpu(), "points must be a CPU tensor");

  const int NDim = voxel_mapping.size(1);
  const int num_points = points.size(0);
  const int num_features = points.size(1);

  std::vector<int> grid_size(NDim);
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }

  at::Tensor num_points_per_voxel = at::zeros(
      {
          num_points,
      },
      voxel_mapping.options());
  at::Tensor coor_to_voxelidx = -at::ones(
      {grid_size[2], grid_size[1], grid_size[0]}, voxel_mapping.options());
  at::Tensor point_to_voxelidx = -at::ones(
      {
          num_points,
      },
      voxel_mapping.options());

  int voxel_num = 0;
  int max_points = 0;
  AT_DISPATCH_ALL_TYPES(voxel_mapping.scalar_type(), "determin_max_point", [&] {
    determin_max_points_kernel<scalar_t>(
        voxel_mapping.accessor<scalar_t, 2>(),
        point_to_voxelidx.accessor<scalar_t, 1>(),
        num_points_per_voxel.accessor<scalar_t, 1>(),
        coor_to_voxelidx.accessor<scalar_t, 3>(), voxel_num, max_points,
        num_points);
  });

  at::Tensor voxels =
      at::zeros({voxel_num, max_points, num_features}, points.options());
  at::Tensor voxel_coors =
      at::zeros({voxel_num, NDim}, points.options().dtype(at::kInt));

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "scatter_point_to_voxel", [&] {
    scatter_point_to_voxel_kernel<scalar_t, int>(
        points.accessor<scalar_t, 2>(), voxel_mapping.accessor<int, 2>(),
        point_to_voxelidx.accessor<int, 1>(),
        coor_to_voxelidx.accessor<int, 3>(), voxels.accessor<scalar_t, 3>(),
        voxel_coors.accessor<int, 2>(), num_features, num_points, NDim);
  });

  at::Tensor num_points_per_voxel_out =
      num_points_per_voxel.slice(/*dim=*/0, /*start=*/0, /*end=*/voxel_num);
  return {voxels, voxel_coors, num_points_per_voxel_out};
}

}  // namespace voxelization
