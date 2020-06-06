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
#include <spconv/maxpool.h>
#include <spconv/mp_helper.h>
#include <tensorview/helper_kernel.cu.h>
#include <tensorview/helper_launch.h>
#include <tensorview/tensorview.h>

#include <chrono>
#include <limits>
#include <type_traits>

namespace spconv {
template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolFwdBlockKernel(T *outFeatures, const T *inFeatures,
                                      const Index *indicesIn,
                                      const Index *indicesOut, int numHot,
                                      int numPlanes) {
  T in, out;
  int ILPStrideY[NumILP];
  Index idxo, idxi;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x; ix < numHot;
       ix += blockDim.x * gridDim.x) {
    {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        in = inFeatures[idxi];
        out = outFeatures[idxo];
        if (in > out) {
          outFeatures[idxo] = in;
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolFwdGenericBlockKernel(T *outFeatures,
                                             const T *inFeatures,
                                             const Index *indicesIn,
                                             const Index *indicesOut,
                                             int numHot, int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  T in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
      RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        in = inFeatures[RI[ilp] + iy];
        out = outFeatures[RO[ilp] + iy];
        if (in > out) {
          outFeatures[RO[ilp] + iy] = in;
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void maxPoolFwdVecBlockKernel(T *outFeatures, const T *inFeatures,
                                         const Index *indicesIn,
                                         const Index *indicesOut, int numHot,
                                         int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(T);
  T bufi[vecloadFactor];
  T bufo[vecloadFactor];
  Index idxi, idxo;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x * vecloadFactor; ix < numHot;
       ix += blockDim.x * gridDim.x * vecloadFactor) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      reinterpret_cast<VecType *>(bufo)[0] =
          reinterpret_cast<VecType *>(outFeatures)[idxo];
      reinterpret_cast<VecType *>(bufi)[0] =
          reinterpret_cast<const VecType *>(inFeatures)[idxi];
#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        if (bufi[i] > bufo[i]) {
          bufo[i] = bufi[i];
        }
      }
      reinterpret_cast<VecType *>(outFeatures)[idxo] =
          reinterpret_cast<VecType *>(bufo)[0];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolFwdGenericKernel(T *outFeatures, const T *inFeatures,
                                        const Index *indicesIn,
                                        const Index *indicesOut, int numHot,
                                        int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  T in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < numHot) {
        RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
        RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
      }
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < numHot) {
          in = inFeatures[RI[ilp] + iy];
          out = outFeatures[RO[ilp] + iy];
          if (in > out) {
            outFeatures[RO[ilp] + iy] = in;
          }
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolBwdBlockKernel(const T *outFeatures, const T *inFeatures,
                                      const T *dout, T *din,
                                      const Index *indicesIn,
                                      const Index *indicesOut, int numHot,
                                      int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  T in, out;
  Index idxo, idxi;
  int ILPStrideY[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  dout += blockIdx.y * NumTLP;
  din += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x; ix < numHot;
       ix += blockDim.x * gridDim.x) {
    {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        in = inFeatures[idxi];
        out = outFeatures[idxo];
        if (in == out) {
          din[idxi] += dout[idxo];
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolBwdGenericBlockKernel(const T *outFeatures,
                                             const T *inFeatures, const T *dout,
                                             T *din, const Index *indicesIn,
                                             const Index *indicesOut,
                                             int numHot, int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  T in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
      RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        in = inFeatures[RI[ilp] + iy];
        out = outFeatures[RO[ilp] + iy];
        if (in == out) {
          din[RI[ilp] + iy] += dout[RO[ilp] + iy];
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void maxPoolBwdVecBlockKernel(const T *outFeatures,
                                         const T *inFeatures, const T *dout,
                                         T *din, const Index *indicesIn,
                                         const Index *indicesOut, int numHot,
                                         int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(T);
  T bufi[vecloadFactor];
  T bufo[vecloadFactor];
  T bufdi[vecloadFactor];
  T bufdo[vecloadFactor];
  Index idxi, idxo;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x * vecloadFactor; ix < numHot;
       ix += blockDim.x * gridDim.x * vecloadFactor) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      reinterpret_cast<VecType *>(bufo)[0] =
          reinterpret_cast<const VecType *>(outFeatures)[idxo];
      reinterpret_cast<VecType *>(bufi)[0] =
          reinterpret_cast<const VecType *>(inFeatures)[idxi];
      reinterpret_cast<VecType *>(bufdo)[0] =
          reinterpret_cast<const VecType *>(dout)[idxo];
      reinterpret_cast<VecType *>(bufdi)[0] =
          reinterpret_cast<VecType *>(din)[idxi];

#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        if (bufi[i] == bufo[i]) {
          bufdi[i] += bufdo[i];
        }
      }
      reinterpret_cast<VecType *>(din)[idxi] =
          reinterpret_cast<VecType *>(bufdi)[0];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolBwdGenericKernel(const T *outFeatures,
                                        const T *inFeatures, const T *dout,
                                        T *din, const Index *indicesIn,
                                        const Index *indicesOut, int numHot,
                                        int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  T in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < numHot) {
        RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
        RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
      }
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < numHot) {
          in = inFeatures[RI[ilp] + iy];
          out = outFeatures[RO[ilp] + iy];
          if (in == out) {
            din[RI[ilp] + iy] += dout[RO[ilp] + iy];
          }
        }
      }
    }
  }
}

namespace functor {
template <typename T, typename Index>
struct SparseMaxPoolForwardFunctor<tv::GPU, T, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::GPU &d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const Index> indices, int size) {
    if (size <= 0) return;
    int numPlanes = inFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(T);
    mp_for_each<kernel_block_t>([=, &outFeatures, &inFeatures, &indices,
                                 &notFound](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;

      int numHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (numHotBlock >= NumTLP) {
            maxPoolFwdVecBlockKernel<T, Index, int(NumTLP), NumILP,
                                     vecload_type_t>
                <<<dim3(std::min(size / NumTLP, 512), numPlanes / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.getStream()>>>(outFeatures.data(), inFeatures.data(),
                                    indices.subview(0).data(),
                                    indices.subview(1).data(), numHotBlock,
                                    numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }

          if (size > numHotBlock) {
            maxPoolFwdGenericKernel<T, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.getStream()>>>(outFeatures.data(), inFeatures.data(),
                                       indices.subview(0).data() + numHotBlock,
                                       indices.subview(1).data() + numHotBlock,
                                       size - numHotBlock, numPlanes);
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });

    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      int numHotBlock = (size / NumTLP) * NumTLP;
      if (numHotBlock >= NumTLP) {
        maxPoolFwdGenericBlockKernel<T, Index, NumTLP, NumILP>
            <<<dim3(size / NumTLP, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
                outFeatures.data(), inFeatures.data(),
                indices.subview(0).data(), indices.subview(1).data(),
                numHotBlock, numPlanes);
        TV_CHECK_CUDA_ERR();
      }

      if (size > numHotBlock) {
        maxPoolFwdGenericKernel<T, Index, NumTLP, NumILP>
            <<<dim3(1, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
                outFeatures.data(), inFeatures.data(),
                indices.subview(0).data() + numHotBlock,
                indices.subview(1).data() + numHotBlock, size - numHotBlock,
                numPlanes);
        TV_CHECK_CUDA_ERR();
      }
    }
  }
};

template <typename T, typename Index>
struct SparseMaxPoolBackwardFunctor<tv::GPU, T, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::GPU &d, tv::TensorView<const T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const T> dout, tv::TensorView<T> din,
                  tv::TensorView<const Index> indices, int size) {
    if (size <= 0) return;
    int numPlanes = inFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(T);
    mp_for_each<kernel_block_t>([=, &outFeatures, &inFeatures, &dout, &din,
                                 &indices, &notFound](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;

      int numHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (numHotBlock >= NumTLP) {
            maxPoolBwdVecBlockKernel<T, Index, int(NumTLP), NumILP,
                                     vecload_type_t>
                <<<dim3(std::min(size / NumTLP, 512), numPlanes / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.getStream()>>>(outFeatures.data(), inFeatures.data(),
                                    dout.data(), din.data(),
                                    indices.subview(0).data(),
                                    indices.subview(1).data(), numHotBlock,
                                    numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }

          if (size > numHotBlock) {
            maxPoolBwdGenericKernel<T, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.getStream()>>>(outFeatures.data(), inFeatures.data(),
                                       dout.data(), din.data(),
                                       indices.subview(0).data() + numHotBlock,
                                       indices.subview(1).data() + numHotBlock,
                                       size - numHotBlock, numPlanes);
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });

    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      int numHotBlock = (size / NumTLP) * NumTLP;
      if (numHotBlock >= NumTLP) {
        maxPoolBwdGenericBlockKernel<T, Index, NumTLP, NumILP>
            <<<dim3(size / NumTLP, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
                outFeatures.data(), inFeatures.data(), dout.data(), din.data(),
                indices.subview(0).data(), indices.subview(1).data(),
                numHotBlock, numPlanes);
        TV_CHECK_CUDA_ERR();
      }

      if (size > numHotBlock) {
        maxPoolBwdGenericKernel<T, Index, NumTLP, NumILP>
            <<<dim3(1, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
                outFeatures.data(), inFeatures.data(), dout.data(), din.data(),
                indices.subview(0).data() + numHotBlock,
                indices.subview(1).data() + numHotBlock, size - numHotBlock,
                numPlanes);
        TV_CHECK_CUDA_ERR();
      }
    }
  }
};

}  // namespace functor

#define DECLARE_GPU_SPECS_T_INDEX(T, Index)                                \
  template struct functor::SparseMaxPoolForwardFunctor<tv::GPU, T, Index>; \
  template struct functor::SparseMaxPoolBackwardFunctor<tv::GPU, T, Index>;

#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPECS_T_INDEX(T, int);

DECLARE_GPU_SPECS(float);
DECLARE_GPU_SPECS(double);
DECLARE_GPU_SPECS(at::Half);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_T_INDEX
}  // namespace spconv
