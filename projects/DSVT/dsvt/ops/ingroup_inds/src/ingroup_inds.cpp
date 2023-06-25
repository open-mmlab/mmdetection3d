#include <assert.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <vector>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


void ingroup_inds_launcher(
    const long *group_inds_data,
    long *out_inds_data,
    int N,
    int max_group_id
);


void ingroup_inds_gpu(
  at::Tensor group_inds,
  at::Tensor out_inds
);

void ingroup_inds_gpu(
  at::Tensor group_inds,
  at::Tensor out_inds
) {

  CHECK_INPUT(group_inds);
  CHECK_INPUT(out_inds);
  int N = group_inds.size(0);
  int max_group_id = group_inds.max().item().toLong();


  long *group_inds_data = group_inds.data_ptr<long>();
  long *out_inds_data = out_inds.data_ptr<long>();

  ingroup_inds_launcher(
      group_inds_data,
      out_inds_data,
      N,
      max_group_id
  );

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ingroup_inds_gpu, "cuda version of get_inner_win_inds of SST");
}
