// Modified from https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/paconv_lib/src/gpu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>


#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))


#define CHECK_CONTIGUOUS(x)                                          \
  do {                                                               \
    AT_ASSERT(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CUDA_CHECK_ERRORS()                                           \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
              __FILE__);                                              \
      exit(-1);                                                       \
    }                                                                 \
  } while (0)


// input: points(B,N0,M,O), centers(B,N0,M,O), scores(B,N1,K,M), knn_idx(B,N1,K)
// output: fout(B,O,N)
// algo: fout(b,i,k,j) = s(b,i,k,m)*p(b,c(i),k,m,j) =  s(b,i,k,m)*p(b,i(k),m,j)
//       i(k) = idx(b,i,k)
//      sum: fout(b,i,j) = fout(b,i,j) + s(b,i,k,m)*p(b,i,k,m,j)
//      avg: fout(b,i,j) = sum(fout(b,i,k,j)) / k
//      max: fout(b,i,j) = max(fout(b,i,k,j), sum(s(b,i,k,m)*p(b,i,k,m,j)))


__global__ void assign_score_withk_forward_kernel(const int B, const int N0, const int N1,
                                                  const int M, const int K, const int O, const int aggregate,
                                                  const float* points,
                                                  const float* centers,
                                                  const float* scores,
                                                  const int64_t* knn_idx,
                                                  float* output) {

    // ----- parallel loop for B, N1, K and O ---------
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B*N1*K*O) return;
    // ------- loop for M ----------
    for (int m = 0; m < M; m++) {
        int b = (int)(i / (O * N1 * K));
        int o = (int)(i % (O * N1 * K) / (N1 * K));
        int n = (int)(i % (N1 * K) / K);
        int k = (int)(i % K);
        int cn = (int) knn_idx[b*K*N1 + n*K + 0]; //The first neighbor is the center point
        int kn = (int) knn_idx[b*K*N1 + n*K + k];
        if (kn >= N0 || kn < 0) { // if index overflows, it is out of the neighborhood range
            continue;
        }
        assert (b < B);
        assert (kn < N0);
        assert (cn < N0);
        assert (o < O);
        assert (n < N1);
        atomicAdd(output + b*N1*O*K + o*N1*K + n*K + k,
            points[b*N0*M*O + kn*M*O + m*O + o] * scores[b*N1*K*M + n*K*M + k*M + m]
                - centers[b*N0*M*O + cn*M*O + m*O + o] * scores[b*N1*K*M + n*K*M + k*M + m]);
    }
}


__global__ void assign_score_withk_backward_points_kernel(const int B, const int N0, const int N, const int M,
                                                          const int K, const int O, const int aggregate,
                                                          const float* grad_out,
                                                          const float* scores,
                                                          const int64_t* knn_idx,
                                                          float* grad_points,
                                                          float* grad_centers) {

    // ----- parallel loop for B, M, O ---------
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B*M*O) return;
    int b = (int)(i / (M * O));
    int m = (int)(i % (M * O) / O);
    int o = (int)(i % O);

    // ----- loop for N,K ---------
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            int kn = knn_idx[b*N*K + n*K + k];
            int cn = knn_idx[b*N*K + n*K + 0];
            if (kn >= N0 || kn < 0) { // if index overflows, it is out of the neighborhood range
                continue;
            }
            atomicAdd(grad_points + b*N0*M*O + kn*M*O + m*O + o,
                scores[b*N*K*M + n*K*M + k*M + m] * grad_out[b*O*N*K + o*N*K + n*K + k]);
            atomicAdd(grad_centers + b*N0*M*O + cn*M*O + m*O + o,
                - scores[b*N*K*M + n*K*M + k*M + m] * grad_out[b*O*N*K + o*N*K + n*K + k]);
            }
    }

}


__global__ void assign_score_withk_backward_scores_kernel(const int B, const int N0, const int N, const int M,
                                                          const int K, const int O, const int aggregate,
                                                          const float* grad_out,
                                                          const float* points,
                                                          const float* centers,
                                                          const int64_t* knn_idx,
                                                          float* grad_scores) {

    // ----- parallel loop for B, N, K, M ---------
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B*N*K*M) return;
    int b = (int)(i / (N * M * K));
    int n = (int)(i % (N * M * K) / M / K);
    int k = (int)(i % (M * K) / M);
    int m = (int)(i % M);
    int cn = knn_idx[b*N*K + n*K + 0];
    int kn = knn_idx[b*N*K + n*K + k];
    if (kn >= N0 || kn < 0) { // if index overflows, it is out of the neighborhood range
        return;
    }

    // -------------- loop for O ------------------------
    for(int o = 0; o < O; o++) {
        atomicAdd(grad_scores + b*N*K*M + n*K*M + k*M + m,
            (points[b*N0*M*O + kn*M*O + m*O + o]
                - centers[b*N0*M*O + cn*M*O + m*O + o])* grad_out[b*O*N*K + o*N*K + n*K + k]);
    }
}


void assign_score_withk_forward_wrapper(int B, int N0, int N1, int M, int K, int O, int aggregate,
                                        const at::Tensor& points,
                                        const at::Tensor& centers,
                                        const at::Tensor& scores,
                                        const at::Tensor& knn_idx,
                                        at::Tensor& output) {
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(centers);
    CHECK_CONTIGUOUS(scores);
    CHECK_CONTIGUOUS(knn_idx);
    CHECK_CONTIGUOUS(output);

    const float* points_data = points.data_ptr<float>();
    const float* centers_data = centers.data_ptr<float>();
    const float* scores_data = scores.data_ptr<float>();
    const int64_t* knn_idx_data = knn_idx.data_ptr<int64_t>();
    float* output_data = output.data_ptr<float>();

    dim3 blocks(DIVUP(B*O*N1*K, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    assign_score_withk_forward_kernel<<<blocks, threads, 0>>>(
        B, N0, N1, M, K, O, aggregate, points_data, centers_data, scores_data, knn_idx_data, output_data);
    CUDA_CHECK_ERRORS();

}


void assign_score_withk_backward_wrapper(int B, int N0, int N1, int M, int K, int O, int aggregate,
                                         const at::Tensor& grad_out,
                                         const at::Tensor& points,
                                         const at::Tensor& centers,
                                         const at::Tensor& scores,
                                         const at::Tensor& knn_idx,
                                         at::Tensor& grad_points,
                                         at::Tensor& grad_centers,
                                         at::Tensor& grad_scores) {

    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(scores);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(centers);
    CHECK_CONTIGUOUS(knn_idx);
    CHECK_CONTIGUOUS(grad_scores);
    CHECK_CONTIGUOUS(grad_points);
    CHECK_CONTIGUOUS(grad_centers);

    const float* grad_out_data = grad_out.data_ptr<float>();
    const float* points_data = points.data_ptr<float>();
    const float* centers_data = centers.data_ptr<float>();
    const float* scores_data = scores.data_ptr<float>();
    const int64_t* knn_idx_data = knn_idx.data_ptr<int64_t>();
    float* grad_points_data = grad_points.data_ptr<float>();
    float* grad_centers_data = grad_centers.data_ptr<float>();
    float* grad_scores_data = grad_scores.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 blocks1(DIVUP(B*M*O, THREADS_PER_BLOCK));
    dim3 threads1(THREADS_PER_BLOCK);
    dim3 blocks2(DIVUP(B*N1*K*M, THREADS_PER_BLOCK));
    dim3 threads2(THREADS_PER_BLOCK);
    assign_score_withk_backward_points_kernel<<<blocks1, threads1, 0>>>(
        B, N0, N1, M, K, O, aggregate, grad_out_data, scores_data, knn_idx_data, grad_points_data, grad_centers_data);
    assign_score_withk_backward_scores_kernel<<<blocks2, threads2, 0>>>(
        B, N0, N1, M, K, O, aggregate, grad_out_data, points_data, centers_data, knn_idx_data, grad_scores_data);

    CUDA_CHECK_ERRORS();
}
