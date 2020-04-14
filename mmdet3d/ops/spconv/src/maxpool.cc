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

#include <spconv/maxpool.h>
#include <torch/script.h>

namespace spconv {

namespace functor {
template <typename T, typename Index>
struct SparseMaxPoolForwardFunctor<tv::CPU, T, Index> {
  void operator()(const tv::CPU &d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const Index> indices, int size) {
    int stride = outFeatures.dim(1);
    auto outFeaturesData = outFeatures.data();
    auto inFeaturesData = inFeatures.data();
    auto indicesIn = indices.subview(0).data();
    auto indicesOut = indices.subview(1).data();
    Index idxi, idxo;
    for (int row = 0; row < size; row++) {
      idxi = indicesIn[row] * stride;
      idxo = indicesOut[row] * stride;
      for (int plane = 0; plane < stride; ++plane)
        if (outFeaturesData[idxo + plane] < inFeaturesData[idxi + plane])
          outFeaturesData[idxo + plane] = inFeaturesData[idxi + plane];
    }
  }
};

template <typename T, typename Index>
struct SparseMaxPoolBackwardFunctor<tv::CPU, T, Index> {
  void operator()(const tv::CPU &d, tv::TensorView<const T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const T> dout, tv::TensorView<T> din,
                  tv::TensorView<const Index> indices, int size) {
    int stride = outFeatures.dim(1);
    auto outFeaturesData = outFeatures.data();
    auto inFeaturesData = inFeatures.data();
    auto doutData = dout.data();
    auto dinData = din.data();
    auto indicesIn = indices.subview(0).data();
    auto indicesOut = indices.subview(1).data();
    Index idxi, idxo;
    for (int row = 0; row < size; row++) {
      idxi = indicesIn[row] * stride;
      idxo = indicesOut[row] * stride;
      for (int plane = 0; plane < stride; ++plane)
        if (outFeaturesData[idxo + plane] == inFeaturesData[idxi + plane])
          dinData[idxi + plane] += doutData[idxo + plane];
    }
  }
};
} // namespace functor

#define DECLARE_CPU_SPECS_T_INDEX(T, Index)                                    \
  template struct functor::SparseMaxPoolForwardFunctor<tv::CPU, T, Index>;     \
  template struct functor::SparseMaxPoolBackwardFunctor<tv::CPU, T, Index>;

#define DECLARE_CPU_SPECS(T)                                                   \
  DECLARE_CPU_SPECS_T_INDEX(T, int);                                           \
  DECLARE_CPU_SPECS_T_INDEX(T, long);

DECLARE_CPU_SPECS(float);
DECLARE_CPU_SPECS(double);
DECLARE_CPU_SPECS(at::Half);

#undef DECLARE_CPU_SPECS
#undef DECLARE_CPU_SPECS_T_INDEX

} // namespace spconv
