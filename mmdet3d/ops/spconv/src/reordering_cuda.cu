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

#include <ATen/ATen.h>
#include <spconv/mp_helper.h>
#include <spconv/reordering.cu.h>
#include <spconv/reordering.h>
#include <tensorview/helper_kernel.cu.h>
#include <tensorview/helper_launch.h>
#include <tensorview/tensorview.h>
#include <utility/timer.h>

#include <chrono>
#include <limits>
#include <type_traits>

namespace spconv {
namespace functor {
template <typename T, typename Index>
struct SparseGatherFunctor<tv::GPU, T, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::GPU &d, tv::TensorView<T> buffer,
                  tv::TensorView<const T> features,
                  tv::TensorView<const Index> indices, int size) {
    if (size <= 0) return;
    int numPlanes = features.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(T);
    mp_for_each<kernel_block_t>([=, &buffer, &features, &indices,
                                 &notFound](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;
      // constexpr int NumILP = NumTLP / (64 / (NumTLP / vecloadFactor));
      int nHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (nHotBlock >= NumTLP) {
            gatherVecBlockKernel<T, Index, int(NumTLP), NumILP, vecload_type_t>
                <<<dim3(numPlanes / NumTLP, size / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.getStream()>>>(buffer.data(), features.data(),
                                    indices.data(), nHotBlock,
                                    numPlanes / vecloadFactor);

            TV_CHECK_CUDA_ERR();
          }
          if (size - nHotBlock > 0) {
            gatherVecKernel<T, Index, int(NumTLP), NumILP, vecload_type_t>
                <<<dim3(1, numPlanes / NumTLP),
                   dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                   d.getStream()>>>(buffer.data() + nHotBlock * numPlanes,
                                    features.data(), indices.data() + nHotBlock,
                                    size - nHotBlock,
                                    numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });

    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      gatherGenericKernel<T, Index, NumTLP, NumILP>
          <<<dim3(tv::launch::DivUp(size, NumTLP),
                  tv::launch::DivUp(numPlanes, NumTLP)),
             dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
              buffer.data(), features.data(), indices.data(), size, numPlanes);
      TV_CHECK_CUDA_ERR();
    }
  }
};
template <typename T, typename Index>
struct SparseScatterAddFunctor<tv::GPU, T, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::GPU &d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> buffer,
                  tv::TensorView<const Index> indices, int size, bool stable) {
    if (size <= 0) return;
    int numPlanes = outFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor =
        sizeof(vecload_type_t) / sizeof(T);  // important for half.
    mp_for_each<kernel_block_t>([=, &d, &outFeatures, &buffer, &indices,
                                 &notFound](auto NumTLP) {
      // constexpr int NumILP = NumTLP / (64 / (NumTLP / vecloadFactor));
      constexpr int NumILP = NumTLP / 4;
      int nHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (nHotBlock >= NumTLP) {
            scatterAddVecBlockKernel<T, Index, int(NumTLP), NumILP,
                                     vecload_type_t>
                <<<dim3(numPlanes / NumTLP, size / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.getStream()>>>(outFeatures.data(), buffer.data(),
                                    indices.data(), nHotBlock,
                                    numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }
          if (size - nHotBlock > 0) {
            scatterAddGenericKernel<T, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.getStream()>>>(
                    outFeatures.data(), buffer.data() + nHotBlock * numPlanes,
                    indices.data() + nHotBlock, size - nHotBlock, numPlanes);
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });
    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      scatterAddGenericKernel<T, Index, NumTLP, NumILP>
          <<<dim3(tv::launch::DivUp(size, NumTLP),
                  tv::launch::DivUp(numPlanes, NumTLP)),
             dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
              outFeatures.data(), buffer.data(), indices.data(), size,
              numPlanes);
      TV_CHECK_CUDA_ERR();
    }
  }
};
}  // namespace functor

#define DECLARE_GPU_SPECS_T_INDEX(T, Index)                        \
  template struct functor::SparseGatherFunctor<tv::GPU, T, Index>; \
  template struct functor::SparseScatterAddFunctor<tv::GPU, T, Index>;

#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPECS_T_INDEX(T, int);

DECLARE_GPU_SPECS(float);
DECLARE_GPU_SPECS(double);
DECLARE_GPU_SPECS(at::Half);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_T_INDEX
}  // namespace spconv
