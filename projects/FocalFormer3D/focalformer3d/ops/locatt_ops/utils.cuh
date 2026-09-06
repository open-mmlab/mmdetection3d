#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cstdio>


#define CUDA_NUM_THREADS 1024
#define MAX_PIXELS_2d 1048576
#define MAX_PIXELS_3d 16777216

#define KERNEL_LOOP(i, I)                              \
for (int i = threadIdx.x; i < (I); i += blockDim.x)

#define KERNEL_LOOP1d(i, I)                            \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (I); i += gridDim.x * blockDim.x)

inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

inline int GET_BLOCKS(const int N, const int Chunks) {
    return (N + Chunks - 1) / Chunks;
}

inline void TypeCheck(const torch::Tensor &input) {
    //AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
}
