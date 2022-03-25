#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "interpolate_gpu.h"


__global__ void three_nn_kernel_fast(int n, int m, const float *__restrict__ unknown,
    const float *__restrict__ known, float *__restrict__ dist2, int *__restrict__ idx) {
    // unknown: (N, 4)
    // known: (M, 4)
    // output:
    //      dist2: (N, 3)
    //      idx: (N, 3)


    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= n) return;

    unknown += pt_idx * 4;

    dist2 += pt_idx * 3;
    idx += pt_idx * 3;

    float ub = unknown[0];
    float ux = unknown[1];
    float uy = unknown[2];
    float uz = unknown[3];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
        float b = known[k * 4 + 0]; //batch number
        if (b!=ub)
            continue;
        float x = known[k * 4 + 1];
        float y = known[k * 4 + 2];
        float z = known[k * 4 + 3];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
        if (d < best1) {
            best3 = best2; besti3 = besti2;
            best2 = best1; besti2 = besti1;
            best1 = d; besti1 = k;
        }
        else if (d < best2) {
            best3 = best2; besti3 = besti2;
            best2 = d; besti2 = k;
        }
        else if (d < best3) {
            best3 = d; besti3 = k;
        }
    }
    dist2[0] = best1; dist2[1] = best2; dist2[2] = best3;
    idx[0] = besti1; idx[1] = besti2; idx[2] = besti3;
}

void three_nn_kernel_launcher_fast(int n, int m, const float *unknown,
    const float *known, float *dist2, int *idx, cudaStream_t stream) {
    // unknown: (N, 4)
    // known: (M, 4)
    // output: 
    //      dist2: (N, 3)
    //      idx: (N, 3)

    cudaError_t err;
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    three_nn_kernel_fast<<<blocks, threads, 0, stream>>>(n, m, unknown, known, dist2, idx);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void three_interpolate_kernel_fast(int c, int m, int n, const float *__restrict__ points,
    const int *__restrict__ idx, const float *__restrict__ weight, float *__restrict__ out) {
    // points: (M, C)
    // idx: (N, 3)
    // weight: (N, 3)
    // output:
    //      out: (N, C)


    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_idx >= c || pt_idx >= n) return;

    weight += pt_idx * 3;
    //points += c_idx * m;

    idx += pt_idx * 3;

    out += pt_idx * c;

    out[c_idx] = weight[0] * points[idx[0] * c + c_idx] + weight[1] * points[idx[1] * c + c_idx] + weight[2] * points[idx[2] * c + c_idx];
}

void three_interpolate_kernel_launcher_fast(int c, int m, int n,
    const float *points, const int *idx, const float *weight, float *out, cudaStream_t stream) {
   // points: (M, C)
    // idx: (N, 3)
    // weight: (N, 3)
    // output:
    //      out: (N, C)

    cudaError_t err;
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    three_interpolate_kernel_fast<<<blocks, threads, 0, stream>>>(c, m, n, points, idx, weight, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void three_interpolate_grad_kernel_fast(int c, int n, int m, const float *__restrict__ grad_out,
    const int *__restrict__ idx, const float *__restrict__ weight, float *__restrict__ grad_points) {
    // grad_out: (N, C)
    // weight: (N, 3)
    // idx: (N, 3)
    // output:
    //      grad_points: (M, C)


    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_idx >= c || pt_idx >= n) return;
    
    grad_out += pt_idx * c + c_idx;
    weight += pt_idx * 3;
    //grad_points += c_idx * m;
    idx += pt_idx * 3;

    atomicAdd(grad_points + idx[0] * c + c_idx, grad_out[0] * weight[0]);
    atomicAdd(grad_points + idx[1] * c + c_idx, grad_out[0] * weight[1]);
    atomicAdd(grad_points + idx[2] * c + c_idx, grad_out[0] * weight[2]);
}

void three_interpolate_grad_kernel_launcher_fast(int c, int n, int m, const float *grad_out,
    const int *idx, const float *weight, float *grad_points, cudaStream_t stream) {
    // grad_out: (N, C)
    // weight: (N, 3)
    // idx: (N, 3)
    // output:
    //      grad_points: (M, C)

    cudaError_t err;
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    three_interpolate_grad_kernel_fast<<<blocks, threads, 0, stream>>>(c, n, m, grad_out, idx, weight, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}