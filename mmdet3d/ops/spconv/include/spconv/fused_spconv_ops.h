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

#ifndef FUSED_SPARSE_CONV_OP_H_
#define FUSED_SPARSE_CONV_OP_H_

#include <cuda_runtime_api.h>
#include <spconv/indice.h>
#include <spconv/reordering.h>
#include <torch/script.h>
#include <torch_utils.h>
#include <utility/timer.h>

namespace spconv {
// torch.jit's doc says only support int64, so we need to convert to int32.

template <typename T>
torch::Tensor fusedIndiceConvBatchNorm(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numActOut,
    int64_t _inverse, int64_t _subM) {
  bool subM = _subM != 0;
  bool inverse = _inverse != 0;
  auto device = features.device().type();
  auto ndim = filters.dim() - 2;
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto numOutPlanes = filters.size(ndim + 1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto indicePairMaxSizeIter =
      std::max_element(indicePairNumCpu.data_ptr<int>(),
                       indicePairNumCpu.data_ptr<int>() + kernelVolume);
  int indicePairMaxOffset =
      indicePairMaxSizeIter - indicePairNumCpu.data_ptr<int>();
  int indicePairMaxSize = *indicePairMaxSizeIter;

  /*if (_subM){
    std::vector<int> indicePairNumVec(indicePairNumCpu.data_ptr<int>(),
  indicePairNumCpu.data_ptr<int>() + kernelVolume);
    indicePairNumVec.erase(indicePairNumVec.begin() + indicePairMaxOffset);

    auto indicePairVecMaxSizeIter = std::max_element(
        indicePairNumVec.begin(), indicePairNumVec.end());
    indicePairMaxSize = *indicePairVecMaxSizeIter;
  }*/

  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  // auto indicePairOptions =
  //     torch::TensorOptions().dtype(torch::kInt64).device(indicePairs.device());

  torch::Tensor output =
      torch::zeros({numActOut, numOutPlanes}, options).copy_(bias);
  torch::Tensor inputBuffer =
      torch::zeros({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::zeros({indicePairMaxSize, numOutPlanes}, options);
  filters = filters.view({-1, numInPlanes, numOutPlanes});
  if (subM) {  // the center index of subm conv don't need gather and scatter
               // add.
    torch::mm_out(output, features, filters[indicePairMaxOffset]);
  }
  double totalGatherTime = 0;
  double totalGEMMTime = 0;
  double totalSAddTime = 0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    // auto timer = spconv::CudaContextTimer<>();
    auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr<T>(),
                                             {nHot, numOutPlanes}, options);
    auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr<T>(),
                                            {nHot, numInPlanes}, options);

    if (device == torch::kCPU) {
      functor::SparseGatherFunctor<tv::CPU, T, int> gatherFtor;
      gatherFtor(tv::CPU(), tv::torch2tv<T>(inputBuffer),
                 tv::torch2tv<const T>(features),
                 tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                 nHot);
    } else {
      functor::SparseGatherFunctor<tv::GPU, T, int> gatherFtor;
      gatherFtor(tv::TorchGPU(), tv::torch2tv<T>(inputBuffer),
                 tv::torch2tv<const T>(features),
                 tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                 nHot);
      TV_CHECK_CUDA_ERR();
      /* slower than SparseGatherFunctor, may due to int->long conversion
      auto indicePairLong = indicePairs[i][inverse].to(torch::kInt64);
      auto indicePairBlob = torch::from_blob(indicePairLong.data_ptr<long>(),
      {nHot}, indicePairOptions); torch::index_select_out(inputBufferBlob,
      features, 0, indicePairBlob);*/
    }
    // totalGatherTime += timer.report() / 1000.0;
    torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);
    // totalGEMMTime += timer.report() / 1000.0;

    if (device == torch::kCPU) {
      functor::SparseScatterAddFunctor<tv::CPU, T, int> scatterFtor;
      scatterFtor(tv::CPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(outputBuffer),
                  tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                  nHot, true);
    } else {
      functor::SparseScatterAddFunctor<tv::GPU, T, int> scatterFtor;
      scatterFtor(tv::TorchGPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(outputBuffer),
                  tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                  nHot, true);
      TV_CHECK_CUDA_ERR();
    }
    // totalSAddTime += timer.report() / 1000.0;
  }
  // std::cout << "gather time " << totalGatherTime << std::endl;
  // std::cout << "gemm time " << totalGEMMTime << std::endl;
  // std::cout << "scatteradd time " << totalSAddTime << std::endl;
  return output;
}
}  // namespace spconv

#endif
