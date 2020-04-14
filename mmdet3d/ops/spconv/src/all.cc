// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime_api.h>
#include <spconv/fused_spconv_ops.h>
#include <spconv/pool_ops.h>
#include <spconv/spconv_ops.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_indice_pairs_2d", &spconv::getIndicePair<2>,
        "get_indice_pairs_2d");
  m.def("get_indice_pairs_3d", &spconv::getIndicePair<3>,
        "get_indice_pairs_3d");
  m.def("get_indice_pairs_4d", &spconv::getIndicePair<4>,
        "get_indice_pairs_4d");
  m.def("get_indice_pairs_grid_2d", &spconv::getIndicePairPreGrid<2>,
        "get_indice_pairs_grid_2d");
  m.def("get_indice_pairs_grid_3d", &spconv::getIndicePairPreGrid<3>,
        "get_indice_pairs_grid_3d");
  m.def("indice_conv_fp32", &spconv::indiceConv<float>, "indice_conv_fp32");
  m.def("indice_conv_backward_fp32", &spconv::indiceConvBackward<float>,
        "indice_conv_backward_fp32");
  m.def("indice_conv_half", &spconv::indiceConv<at::Half>, "indice_conv_half");
  m.def("indice_conv_backward_half", &spconv::indiceConvBackward<at::Half>,
        "indice_conv_backward_half");
  m.def("fused_indice_conv_fp32", &spconv::fusedIndiceConvBatchNorm<float>,
        "fused_indice_conv_fp32");
  m.def("fused_indice_conv_half", &spconv::fusedIndiceConvBatchNorm<at::Half>,
        "fused_indice_conv_half");
  m.def("indice_maxpool_fp32", &spconv::indiceMaxPool<float>,
        "indice_maxpool_fp32");
  m.def("indice_maxpool_backward_fp32", &spconv::indiceMaxPoolBackward<float>,
        "indice_maxpool_backward_fp32");
  m.def("indice_maxpool_half", &spconv::indiceMaxPool<at::Half>,
        "indice_maxpool_half");
  m.def("indice_maxpool_backward_half",
        &spconv::indiceMaxPoolBackward<at::Half>,
        "indice_maxpool_backward_half");
}
