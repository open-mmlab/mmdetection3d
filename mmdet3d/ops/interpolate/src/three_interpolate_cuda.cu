// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void three_interpolate_kernel(int b, int c, int m, int n,
                                         const float *__restrict__ points,
                                         const int *__restrict__ idx,
                                         const float *__restrict__ weight,
                                         float *__restrict__ out) {
  // points: (B, C, M)
  // idx: (B, N, 3)
  // weight: (B, N, 3)
  // output:
  //      out: (B, C, N)

  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

  weight += bs_idx * n * 3 + pt_idx * 3;
  points += bs_idx * c * m + c_idx * m;
  idx += bs_idx * n * 3 + pt_idx * 3;
  out += bs_idx * c * n + c_idx * n;

  out[pt_idx] = weight[0] * points[idx[0]] + weight[1] * points[idx[1]] +
                weight[2] * points[idx[2]];
}

void three_interpolate_kernel_launcher(int b, int c, int m, int n,
                                       const float *points, const int *idx,
                                       const float *weight, float *out,
                                       cudaStream_t stream) {
  // points: (B, C, M)
  // idx: (B, N, 3)
  // weight: (B, N, 3)
  // output:
  //      out: (B, C, N)

  cudaError_t err;
  dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c,
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);
  three_interpolate_kernel<<<blocks, threads, 0, stream>>>(b, c, m, n, points,
                                                           idx, weight, out);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void three_interpolate_grad_kernel(
    int b, int c, int n, int m, const float *__restrict__ grad_out,
    const int *__restrict__ idx, const float *__restrict__ weight,
    float *__restrict__ grad_points) {
  // grad_out: (B, C, N)
  // weight: (B, N, 3)
  // output:
  //      grad_points: (B, C, M)

  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

  grad_out += bs_idx * c * n + c_idx * n + pt_idx;
  weight += bs_idx * n * 3 + pt_idx * 3;
  grad_points += bs_idx * c * m + c_idx * m;
  idx += bs_idx * n * 3 + pt_idx * 3;

  atomicAdd(grad_points + idx[0], grad_out[0] * weight[0]);
  atomicAdd(grad_points + idx[1], grad_out[0] * weight[1]);
  atomicAdd(grad_points + idx[2], grad_out[0] * weight[2]);
}

void three_interpolate_grad_kernel_launcher(int b, int c, int n, int m,
                                            const float *grad_out,
                                            const int *idx, const float *weight,
                                            float *grad_points,
                                            cudaStream_t stream) {
  // grad_out: (B, C, N)
  // weight: (B, N, 3)
  // output:
  //      grad_points: (B, C, M)

  cudaError_t err;
  dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c,
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);
  three_interpolate_grad_kernel<<<blocks, threads, 0, stream>>>(
      b, c, n, m, grad_out, idx, weight, grad_points);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
