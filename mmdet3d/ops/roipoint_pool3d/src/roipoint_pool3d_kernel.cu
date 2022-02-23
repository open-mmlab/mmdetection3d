/*
Modified from
https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d_kernel.cu
Point cloud feature pooling
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
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
  // param box3d: (cx, cy, cz, dx, dy, dz, rz) in LiDAR coordinate, cz in the
  // bottom center
  float x = pt[0], y = pt[1], z = pt[2];
  float cx = box3d[0], cy = box3d[1], cz = box3d[2];
  float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];
  cz += dz / 2.0;  // shift to the center since cz in box3d is the bottom center

  if (fabsf(z - cz) > dz / 2.0) return 0;
  lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
  float in_flag = (local_x > -dx / 2.0) & (local_x < dx / 2.0) &
                  (local_y > -dy / 2.0) & (local_y < dy / 2.0);
  return in_flag;
}

__global__ void assign_pts_to_box3d(int batch_size, int pts_num, int boxes_num, const float *xyz, const float *boxes3d, int *pts_assign){
    // params xyz: (B, N, 3)
    // params boxes3d: (B, M, 7)
    // params pts_assign: (B, N, M): idx of the corresponding box3d, -1 means background points
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    int bs_idx = blockIdx.z;

    if (pt_idx >= pts_num || box_idx >= boxes_num || bs_idx >= batch_size){
        return;
    }
    int assign_idx = bs_idx * pts_num * boxes_num + pt_idx * boxes_num + box_idx;
    pts_assign[assign_idx] = 0;

    int box_offset = bs_idx * boxes_num * 7 + box_idx * 7;
    int pt_offset = bs_idx * pts_num * 3 + pt_idx * 3;


    float local_x = 0, local_y = 0;
    int cur_in_flag = check_pt_in_box3d(xyz + pt_offset, boxes3d + box_offset, local_x, local_y);
    pts_assign[assign_idx] = cur_in_flag;
    // printf("bs=%d, pt=%d, in=%d\n", bs_idx, pt_idx, pts_assign[bs_idx * pts_num + pt_idx]);
}


__global__ void get_pooled_idx(int batch_size, int pts_num, int boxes_num, int sampled_pts_num,
                               const int *pts_assign, int *pts_idx, int *pooled_empty_flag){
    // params xyz: (B, N, 3)
    // params pts_feature: (B, N, C)
    // params pts_assign: (B, N)
    // params pts_idx: (B, M, 512)
    // params pooled_empty_flag: (B, M)

    int boxes_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boxes_idx >= boxes_num){
        return;
    }

    int bs_idx = blockIdx.y;

    int cnt = 0;
    for (int k = 0; k < pts_num; k++){
        if (pts_assign[bs_idx * pts_num * boxes_num + k * boxes_num + boxes_idx]){
            if (cnt < sampled_pts_num){
                pts_idx[bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num + cnt] = k;
                cnt++;
            }
            else break;
        }
    }

    if (cnt == 0){
        pooled_empty_flag[bs_idx * boxes_num + boxes_idx] = 1;
    }
    else if (cnt < sampled_pts_num){
        // duplicate same points for sampling
        for (int k = cnt; k < sampled_pts_num; k++){
            int duplicate_idx = k % cnt;
            int base_offset = bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num;
            pts_idx[base_offset + k] = pts_idx[base_offset + duplicate_idx];
        }
    }
}


__global__ void roipool3d_forward(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
                                   const float *xyz, const int *pts_idx, const float *pts_feature,
                                   float *pooled_features, int *pooled_empty_flag){
    // params xyz: (B, N, 3)
    // params pts_idx: (B, M, 512)
    // params pts_feature: (B, N, C)
    // params pooled_features: (B, M, 512, 3+C)
    // params pooled_empty_flag: (B, M)

    int sample_pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    int bs_idx = blockIdx.z;

    if (sample_pt_idx >= sampled_pts_num || box_idx >= boxes_num || bs_idx >= batch_size){
        return;
    }

    if (pooled_empty_flag[bs_idx * boxes_num + box_idx]){
        return;
    }

    int temp_idx = bs_idx * boxes_num * sampled_pts_num + box_idx * sampled_pts_num + sample_pt_idx;
    int src_pt_idx = pts_idx[temp_idx];
    int dst_feature_offset = temp_idx * (3 + feature_in_len);

    for (int j = 0; j < 3; j++)
        pooled_features[dst_feature_offset + j] = xyz[bs_idx * pts_num * 3 + src_pt_idx * 3 + j];

    int src_feature_offset = bs_idx * pts_num * feature_in_len + src_pt_idx * feature_in_len;
    for (int j = 0; j < feature_in_len; j++)
        pooled_features[dst_feature_offset + 3 + j] = pts_feature[src_feature_offset + j];
}


void roipool3dLauncher(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
                       const float *xyz, const float *boxes3d, const float *pts_feature, float *pooled_features, int *pooled_empty_flag){

    // printf("batch_size=%d, pts_num=%d, boxes_num=%d\n", batch_size, pts_num, boxes_num);
    int *pts_assign = NULL;
    cudaMalloc(&pts_assign, batch_size * pts_num * boxes_num * sizeof(int));  // (batch_size, N, M)
    // cudaMemset(&pts_assign, -1, batch_size * pts_num * boxes_num * sizeof(int));

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    assign_pts_to_box3d<<<blocks, threads>>>(batch_size, pts_num, boxes_num, xyz, boxes3d, pts_assign);

    int *pts_idx = NULL;
    cudaMalloc(&pts_idx, batch_size * boxes_num * sampled_pts_num * sizeof(int));  // (batch_size, M, sampled_pts_num)

    dim3 blocks2(DIVUP(boxes_num, THREADS_PER_BLOCK), batch_size);  // blockIdx.x(col), blockIdx.y(row)
    get_pooled_idx<<<blocks2, threads>>>(batch_size, pts_num, boxes_num, sampled_pts_num, pts_assign, pts_idx, pooled_empty_flag);

    dim3 blocks_pool(DIVUP(sampled_pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);
    roipool3d_forward<<<blocks_pool, threads>>>(batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
                                                      xyz, pts_idx, pts_feature, pooled_features, pooled_empty_flag);

    cudaFree(pts_assign);
    cudaFree(pts_idx);

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}
