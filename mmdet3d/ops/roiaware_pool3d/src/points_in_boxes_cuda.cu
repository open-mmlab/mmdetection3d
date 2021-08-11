// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
// Written by Shaoshuai Shi
// All Rights Reserved 2019.

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <torch/serialize/tensor.h>
#include <torch/types.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
// #define DEBUG

__device__ inline void lidar_to_local_coords(float shift_x, float shift_y,
                                             float rz, float &local_x,
                                             float &local_y) {
  float cosa = cos(-rz), sina = sin(-rz);
  local_x = shift_x * cosa + shift_y * (-sina);
  local_y = shift_x * sina + shift_y * cosa;
}

__device__ inline int check_pt_in_box3d(const float *pt, const float *box3d,
                                        float &local_x, float &local_y) {
  // param pt: (x, y, z)
  // param box3d: (cx, cy, cz, x_size, y_size, z_size, rz) in LiDAR coordinate, cz in the
  // bottom center
  float x = pt[0], y = pt[1], z = pt[2];
  float cx = box3d[0], cy = box3d[1], cz = box3d[2];
  float x_size = box3d[3], y_size = box3d[4], z_size = box3d[5], rz = box3d[6];
  cz += z_size / 2.0;  // shift to the center since cz in box3d is the bottom center

  if (fabsf(z - cz) > z_size / 2.0) return 0;
  lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
  float in_flag = (local_x > -x_size / 2.0) & (local_x < x_size / 2.0) &
                  (local_y > -y_size / 2.0) & (local_y < y_size / 2.0);
  return in_flag;
}

__global__ void points_in_boxes_part_kernel(int batch_size, int boxes_num,
                                            int pts_num, const float *boxes,
                                            const float *pts,
                                            int *box_idx_of_points) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR coordinate, z is
  // the bottom center, each box DO NOT overlaps params pts: (B, npoints, 3) [x,
  // y, z] in LiDAR coordinate params boxes_idx_of_points: (B, npoints), default
  // -1

  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= batch_size || pt_idx >= pts_num) return;

  boxes += bs_idx * boxes_num * 7;
  pts += bs_idx * pts_num * 3 + pt_idx * 3;
  box_idx_of_points += bs_idx * pts_num + pt_idx;

  float local_x = 0, local_y = 0;
  int cur_in_flag = 0;
  for (int k = 0; k < boxes_num; k++) {
    cur_in_flag = check_pt_in_box3d(pts, boxes + k * 7, local_x, local_y);
    if (cur_in_flag) {
      box_idx_of_points[0] = k;
      break;
    }
  }
}

__global__ void points_in_boxes_all_kernel(int batch_size, int boxes_num,
                                           int pts_num, const float *boxes,
                                           const float *pts,
                                           int *box_idx_of_points) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR coordinate, z is
  // the bottom center, each box DO NOT overlaps params pts: (B, npoints, 3) [x,
  // y, z] in LiDAR coordinate params boxes_idx_of_points: (B, npoints), default
  // -1

  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= batch_size || pt_idx >= pts_num) return;

  boxes += bs_idx * boxes_num * 7;
  pts += bs_idx * pts_num * 3 + pt_idx * 3;
  box_idx_of_points += bs_idx * pts_num * boxes_num + pt_idx * boxes_num;

  float local_x = 0, local_y = 0;
  int cur_in_flag = 0;
  for (int k = 0; k < boxes_num; k++) {
    cur_in_flag = check_pt_in_box3d(pts, boxes + k * 7, local_x, local_y);
    if (cur_in_flag) {
      box_idx_of_points[k] = 1;
    }
    cur_in_flag = 0;
  }
}

void points_in_boxes_part_launcher(int batch_size, int boxes_num, int pts_num,
                                   const float *boxes, const float *pts,
                                   int *box_idx_of_points) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR coordinate, z is
  // the bottom center, each box DO NOT overlaps params pts: (B, npoints, 3) [x,
  // y, z] in LiDAR coordinate params boxes_idx_of_points: (B, npoints), default
  // -1
  cudaError_t err;

  dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), batch_size);
  dim3 threads(THREADS_PER_BLOCK);
  points_in_boxes_part_kernel<<<blocks, threads>>>(batch_size, boxes_num, pts_num,
                                                   boxes, pts, box_idx_of_points);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

#ifdef DEBUG
  cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

void points_in_boxes_all_launcher(int batch_size, int boxes_num, int pts_num,
                                  const float *boxes, const float *pts,
                                  int *box_idx_of_points) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR coordinate, z is
  // the bottom center, each box params pts: (B, npoints, 3) [x, y, z] in
  // LiDAR coordinate params boxes_idx_of_points: (B, npoints), default -1
  cudaError_t err;

  dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), batch_size);
  dim3 threads(THREADS_PER_BLOCK);
  points_in_boxes_all_kernel<<<blocks, threads>>>(
      batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

#ifdef DEBUG
  cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

int points_in_boxes_part(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                         at::Tensor box_idx_of_points_tensor) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR coordinate, z is
  // the bottom center, each box DO NOT overlaps params pts: (B, npoints, 3) [x,
  // y, z] in LiDAR coordinate params boxes_idx_of_points: (B, npoints), default
  // -1

  CHECK_INPUT(boxes_tensor);
  CHECK_INPUT(pts_tensor);
  CHECK_INPUT(box_idx_of_points_tensor);

  int batch_size = boxes_tensor.size(0);
  int boxes_num = boxes_tensor.size(1);
  int pts_num = pts_tensor.size(1);

  const float *boxes = boxes_tensor.data_ptr<float>();
  const float *pts = pts_tensor.data_ptr<float>();
  int *box_idx_of_points = box_idx_of_points_tensor.data_ptr<int>();

  points_in_boxes_part_launcher(batch_size, boxes_num, pts_num, boxes, pts,
                                box_idx_of_points);

  return 1;
}

int points_in_boxes_all(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                        at::Tensor box_idx_of_points_tensor) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR coordinate, z is
  // the bottom center. params pts: (B, npoints, 3) [x, y, z] in LiDAR
  // coordinate params boxes_idx_of_points: (B, npoints), default -1

  CHECK_INPUT(boxes_tensor);
  CHECK_INPUT(pts_tensor);
  CHECK_INPUT(box_idx_of_points_tensor);

  int batch_size = boxes_tensor.size(0);
  int boxes_num = boxes_tensor.size(1);
  int pts_num = pts_tensor.size(1);

  const float *boxes = boxes_tensor.data_ptr<float>();
  const float *pts = pts_tensor.data_ptr<float>();
  int *box_idx_of_points = box_idx_of_points_tensor.data_ptr<int>();

  points_in_boxes_all_launcher(batch_size, boxes_num, pts_num, boxes, pts,
                               box_idx_of_points);

  return 1;
}
