// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void ball_query_kernel(int b, int n, int m,
                                  float min_radius,
                                  float max_radius,
                                  int nsample,
                                  const float *__restrict__ new_xyz,
                                  const float *__restrict__ xyz,
                                  int *__restrict__ idx) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= m) return;

  new_xyz += bs_idx * m * 3 + pt_idx * 3;
  xyz += bs_idx * n * 3;
  idx += bs_idx * m * nsample + pt_idx * nsample;

  float max_radius2 = max_radius * max_radius;
  float min_radius2 = min_radius * min_radius;
  float new_x = new_xyz[0];
  float new_y = new_xyz[1];
  float new_z = new_xyz[2];

  int cnt = 0;
  for (int k = 0; k < n; ++k) {
    float x = xyz[k * 3 + 0];
    float y = xyz[k * 3 + 1];
    float z = xyz[k * 3 + 2];
    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
               (new_z - z) * (new_z - z);
    if (d2 == 0 || (d2 >= min_radius2 && d2 < max_radius2)) {
      if (cnt == 0) {
        for (int l = 0; l < nsample; ++l) {
          idx[l] = k;
        }
      }
      idx[cnt] = k;
      ++cnt;
      if (cnt >= nsample) break;
    }
  }
}

void ball_query_kernel_launcher(int b, int n, int m, float min_radius, float max_radius,
                                int nsample, const float *new_xyz, const float *xyz,
                                int *idx, cudaStream_t stream) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)

  cudaError_t err;

  dim3 blocks(DIVUP(m, THREADS_PER_BLOCK),
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  ball_query_kernel<<<blocks, threads, 0, stream>>>(b, n, m, min_radius, max_radius,
                                                    nsample, new_xyz, xyz, idx);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
