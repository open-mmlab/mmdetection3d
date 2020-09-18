#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void feat_distance_kernel(int b, int n, int m, int c,
                                     const float *__restrict__ feat_a,
                                     const float *__restrict__ feat_b,
                                     float *__restrict__ distance) {
  // feat_a: (B, N, C)
  // feat_b: (B, M, C)
  // output:
  //      distance: (B, N, M)
  int bs_idx = blockIdx.y;
  int feat_a_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || feat_a_idx >= n) return;

  feat_a += bs_idx * n * c + feat_a_idx * c;
  feat_b += bs_idx * m * c;
  distance += bs_idx * n * m + feat_a_idx * m;

  const float *cur_feat_b;

  for (int i = 0; i < m; ++i) {
    cur_feat_b = feat_b + i * c;
    for (int j = 0; j < c; ++j) {
      distance[i] += (feat_a[j] - cur_feat_b[j]) * (feat_a[j] - cur_feat_b[j]);
    }
  }
}

void feat_distance_kernel_launcher(int b, int n, int m, int c,
                                   const float *feat_a, const float *feat_b,
                                   float *distance, cudaStream_t stream) {
  // feat_a: (B, N, C)
  // feat_b: (B, M, C)
  // output:
  //      distance: (B, N, M)

  cudaError_t err;

  dim3 blocks(DIVUP(n, THREADS_PER_BLOCK),
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  feat_distance_kernel<<<blocks, threads, 0, stream>>>(b, n, m, c, feat_a,
                                                       feat_b, distance);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
