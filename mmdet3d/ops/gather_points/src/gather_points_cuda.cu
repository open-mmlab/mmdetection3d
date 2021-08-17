#include <stdio.h>
#include <stdlib.h>

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void gather_points_kernel(int b, int c, int n, int m,
                                     const float *__restrict__ points,
                                     const int *__restrict__ idx,
                                     float *__restrict__ out) {
  // points: (B, C, N)
  // idx: (B, M)
  // output:
  //      out: (B, C, M)

  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

  out += bs_idx * c * m + c_idx * m + pt_idx;
  idx += bs_idx * m + pt_idx;
  points += bs_idx * c * n + c_idx * n;
  out[0] = points[idx[0]];
}

void gather_points_kernel_launcher(int b, int c, int n, int npoints,
                                   const float *points, const int *idx,
                                   float *out, cudaStream_t stream) {
  // points: (B, C, N)
  // idx: (B, npoints)
  // output:
  //      out: (B, C, npoints)

  cudaError_t err;
  dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c,
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  gather_points_kernel<<<blocks, threads, 0, stream>>>(b, c, n, npoints, points,
                                                       idx, out);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
                                          const float *__restrict__ grad_out,
                                          const int *__restrict__ idx,
                                          float *__restrict__ grad_points) {
  // grad_out: (B, C, M)
  // idx: (B, M)
  // output:
  //      grad_points: (B, C, N)

  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

  grad_out += bs_idx * c * m + c_idx * m + pt_idx;
  idx += bs_idx * m + pt_idx;
  grad_points += bs_idx * c * n + c_idx * n;

  atomicAdd(grad_points + idx[0], grad_out[0]);
}

void gather_points_grad_kernel_launcher(int b, int c, int n, int npoints,
                                        const float *grad_out, const int *idx,
                                        float *grad_points,
                                        cudaStream_t stream) {
  // grad_out: (B, C, npoints)
  // idx: (B, npoints)
  // output:
  //      grad_points: (B, C, N)

  cudaError_t err;
  dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c,
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  gather_points_grad_kernel<<<blocks, threads, 0, stream>>>(
      b, c, n, npoints, grad_out, idx, grad_points);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
