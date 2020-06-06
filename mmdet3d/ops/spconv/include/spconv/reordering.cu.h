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

#ifndef REORDERING_CU_H_
#define REORDERING_CU_H_
#include <tensorview/helper_kernel.cu.h>

// see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
namespace spconv {
template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void gatherGenericKernel(T *buffer, const T *features,
                                    const Index *indices, int size,
                                    int numPlanes) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size)
          buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
              features[inds[ilp] + iy];
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void gatherVecKernel(T *buffer, const T *features,
                                const Index *indices, int size, int numPlanes) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size)
          reinterpret_cast<VecType *>(
              buffer)[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
              reinterpret_cast<const VecType *>(features)[inds[ilp] + iy];
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP,
          typename VecType = int4>
__global__ void gatherVecBlockKernel(T *buffer, const T *features,
                                     const Index *indices, int size,
                                     int numPlanes) {
  int ILPStrideY[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = ilp * gridDim.y * blockDim.y;
  features += blockIdx.x * NumTLP;
  buffer += blockIdx.x * NumTLP;

  for (int iy : tv::KernelLoopY<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      reinterpret_cast<VecType *>(
          buffer)[(iy + ILPStrideY[ilp]) * numPlanes + threadIdx.x] =
          reinterpret_cast<const VecType *>(
              features)[indices[iy + ILPStrideY[ilp]] * numPlanes +
                        threadIdx.x];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void scatterAddGenericKernel(T *outFeatures, const T *buffer,
                                        const Index *indices, int size,
                                        int numPlanes) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size) {
          outFeatures[inds[ilp] + iy] +=
              buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy];
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP,
          typename VecType = int4>
__global__ void scatterAddVecBlockKernel(T *outFeatures, const T *buffer,
                                         const Index *indices, int size,
                                         int numPlanes) {
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(T);
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = ilp * gridDim.y * blockDim.y;
  outFeatures += blockIdx.x * NumTLP;
  buffer += blockIdx.x * NumTLP;
  T buf[vecloadFactor];
  T buf2[vecloadFactor];
  Index idx;
  for (int iy : tv::KernelLoopY<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idx = indices[iy + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      reinterpret_cast<VecType *>(buf)[0] =
          reinterpret_cast<VecType *>(outFeatures)[idx];
      reinterpret_cast<VecType *>(buf2)[0] = reinterpret_cast<const VecType *>(
          buffer)[(iy + ILPStrideY[ilp]) * numPlanes + threadIdx.x];
#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        buf[i] += buf2[i];
      }
      reinterpret_cast<VecType *>(outFeatures)[idx] =
          reinterpret_cast<VecType *>(buf)[0];
    }
  }
}

}  // namespace spconv

#endif
