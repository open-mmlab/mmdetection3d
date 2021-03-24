// Modified from https://github.com/unlimblue/KNN_CUDA

#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, t) AT_ASSERTM(x.dtype() == t, #x " must be " #t)
#define CHECK_CUDA(x) AT_ASSERTM(x.device().type() == at::Device::Type::CUDA, #x " must be on CUDA")
#define CHECK_INPUT(x, t) CHECK_CONTIGUOUS(x); CHECK_TYPE(x, t); CHECK_CUDA(x)


void knn_kernels_launcher(
    const float* ref_dev,
    int ref_nb,
    const float* query_dev,
    int query_nb,
    int dim,
    int k,
    float* dist_dev,
    long* ind_dev,
    cudaStream_t stream
    );

// std::vector<at::Tensor> knn_wrapper(
void knn_wrapper(
    at::Tensor & ref,
    int ref_nb,
    at::Tensor & query,
    int query_nb,
    at::Tensor & ind,
    const int k
    ) {

    CHECK_INPUT(ref, at::kFloat);
    CHECK_INPUT(query, at::kFloat);
    const float * ref_dev = ref.data_ptr<float>();
    const float * query_dev = query.data_ptr<float>();
    int dim = query.size(0);
    auto dist = at::empty({ref_nb, query_nb}, query.options().dtype(at::kFloat));
    float * dist_dev = dist.data_ptr<float>();
    long * ind_dev = ind.data_ptr<long>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    knn_kernels_launcher(
        ref_dev,
        ref_nb,
        query_dev,
        query_nb,
        dim,
        k,
        dist_dev,
        ind_dev,
        stream
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_wrapper", &knn_wrapper, "knn_wrapper");
}
