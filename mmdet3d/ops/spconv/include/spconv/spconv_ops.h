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

#ifndef SPARSE_CONV_OP_H_
#define SPARSE_CONV_OP_H_

#include <cuda_runtime_api.h>
#include <spconv/indice.h>
#include <spconv/reordering.h>
#include <torch/script.h>
#include <torch_utils.h>
#include <utility/timer.h>

namespace spconv {
// torch.jit's doc says only support int64, so we need to convert to int32.
template <unsigned NDim>
std::vector<torch::Tensor> getIndicePair(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  // auto timer = spconv::CudaContextTimer<>();
  bool subM = _subM != 0;
  bool transpose = _transpose != 0;
  auto numAct = indices.size(0);
  auto coorDim = indices.size(1) - 1;  // batchIdx + xyz
  TV_ASSERT_RT_ERR(NDim == coorDim, "error");
  TV_ASSERT_RT_ERR(kernelSize.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(outSpatialShape.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(stride.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(padding.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(outPadding.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(dilation.size() == coorDim, "error");
  auto kernelVolume = kernelSize[0];
  for (int i = 1; i < kernelSize.size(); ++i) {
    kernelVolume *= kernelSize[i];
  }
  TV_ASSERT_RT_ERR(kernelVolume <= 4096, "error");
  auto outputVolume = outSpatialShape[0];
  for (int i = 1; i < outSpatialShape.size(); ++i) {
    outputVolume *= outSpatialShape[i];
  }
  torch::Tensor indicePairs =
      torch::full({kernelVolume, 2, numAct}, -1,
                  torch::dtype(torch::kInt32).device(indices.device()));
  torch::Tensor indiceNum = torch::zeros(
      {kernelVolume}, torch::dtype(torch::kInt32).device(indices.device()));
  torch::Tensor gridOut =
      torch::full({batchSize * outputVolume}, -1,
                  torch::dtype(torch::kInt32).device(indices.device()));
  // std::cout << "full time " << timer.report() / 1000.0 << std::endl;
  int64_t numActOut = -1;
  tv::SimpleVector<int, NDim> outSpatialShape32;
  tv::SimpleVector<int, NDim> kernelSize32;
  tv::SimpleVector<int, NDim> stride32;
  tv::SimpleVector<int, NDim> padding32;
  tv::SimpleVector<int, NDim> dilation32;
  auto indicePairUnique = torch::full(
      {indicePairs.numel() / 2 + 1}, std::numeric_limits<int>::max(),
      torch::dtype(torch::kInt32).device(indices.device()));
  for (int i = 0; i < NDim; ++i) {
    outSpatialShape32.push_back(outSpatialShape[i]);
    kernelSize32.push_back(kernelSize[i]);
    if (subM) {
      stride32.push_back(1);
      padding32.push_back(kernelSize[i] / 2);
      dilation32.push_back(dilation[i]);
    } else {
      stride32.push_back(stride[i]);
      padding32.push_back(padding[i]);
      dilation32.push_back(dilation[i]);
    }
  }
  if (subM) {
    if (indices.device().type() == torch::kCPU) {
      auto getIndicePairFtor =
          functor::CreateSubMIndicePairFunctor<tv::CPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::CPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(gridOut), tv::torch2tv<int>(indicePairs),
          tv::torch2tv<int>(indiceNum), kernelSize32, stride32, padding32,
          dilation32, outSpatialShape32, transpose);
    } else {
      auto getIndicePairFtor =
          functor::CreateSubMIndicePairFunctor<tv::GPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::TorchGPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(gridOut), tv::torch2tv<int>(indicePairs),
          tv::torch2tv<int>(indiceNum), kernelSize32, stride32, padding32,
          dilation32, outSpatialShape32, transpose);
    }
    return {indices, indicePairs, indiceNum};
  } else {
    torch::Tensor outInds =
        torch::zeros({numAct * kernelVolume, coorDim + 1},
                     torch::dtype(torch::kInt32).device(indices.device()));
    if (indices.device().type() == torch::kCPU) {
      auto getIndicePairFtor =
          functor::CreateConvIndicePairFunctor<tv::CPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::CPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
          tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
          kernelSize32, stride32, padding32, dilation32, outSpatialShape32,
          transpose);
    } else {
      auto getIndicePairFtorP1 =
          functor::CreateConvIndicePairFunctorP1<tv::GPU, int, int, NDim>();
      auto getIndicePairFtorP2 =
          functor::CreateConvIndicePairFunctorP2<tv::GPU, int, int, NDim>();
      numActOut = getIndicePairFtorP1(
          tv::TorchGPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
          tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
          tv::torch2tv<int>(indicePairUnique), kernelSize32, stride32,
          padding32, dilation32, outSpatialShape32, transpose);
      if (numActOut > 0) {
        auto res = torch::_unique(indicePairUnique);
        indicePairUnique = std::get<0>(res);
        numActOut = getIndicePairFtorP2(
            tv::TorchGPU(), tv::torch2tv<const int>(indices),
            tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
            tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
            tv::torch2tv<int>(indicePairUnique), outSpatialShape32, transpose);
      }
    }
    return {outInds.slice(0, 0, numActOut), indicePairs, indiceNum};
  }
}

template <unsigned NDim>
std::vector<torch::Tensor> getIndicePairPreGrid(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  // auto timer = spconv::CudaContextTimer<>();
  bool subM = _subM != 0;
  bool transpose = _transpose != 0;
  auto numAct = indices.size(0);
  auto coorDim = indices.size(1) - 1;  // batchIdx + xyz
  TV_ASSERT_RT_ERR(NDim == coorDim, "error");
  TV_ASSERT_RT_ERR(kernelSize.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(outSpatialShape.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(stride.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(padding.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(outPadding.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(dilation.size() == coorDim, "error");
  auto kernelVolume = kernelSize[0];
  for (int i = 1; i < kernelSize.size(); ++i) {
    kernelVolume *= kernelSize[i];
  }
  TV_ASSERT_RT_ERR(kernelVolume <= 4096, "error");
  auto outputVolume = outSpatialShape[0];
  for (int i = 1; i < outSpatialShape.size(); ++i) {
    outputVolume *= outSpatialShape[i];
  }
  TV_ASSERT_INVALID_ARG(gridOut.numel() >= outputVolume * batchSize, "error");
  torch::Tensor indicePairs =
      torch::full({kernelVolume, 2, numAct}, -1,
                  torch::dtype(torch::kInt32).device(indices.device()));
  torch::Tensor indiceNum = torch::zeros(
      {kernelVolume}, torch::dtype(torch::kInt32).device(indices.device()));
  // std::cout << "full time " << timer.report() / 1000.0 << std::endl;
  int64_t numActOut = -1;
  tv::SimpleVector<int, NDim> outSpatialShape32;
  tv::SimpleVector<int, NDim> kernelSize32;
  tv::SimpleVector<int, NDim> stride32;
  tv::SimpleVector<int, NDim> padding32;
  tv::SimpleVector<int, NDim> dilation32;
  auto indicePairUnique = torch::full(
      {indicePairs.numel() / 2 + 1}, std::numeric_limits<int>::max(),
      torch::dtype(torch::kInt32).device(indices.device()));
  for (int i = 0; i < NDim; ++i) {
    outSpatialShape32.push_back(outSpatialShape[i]);
    kernelSize32.push_back(kernelSize[i]);
    if (subM) {
      stride32.push_back(1);
      padding32.push_back(kernelSize[i] / 2);
      dilation32.push_back(dilation[i]);
    } else {
      stride32.push_back(stride[i]);
      padding32.push_back(padding[i]);
      dilation32.push_back(dilation[i]);
    }
  }
  if (subM) {
    if (indices.device().type() == torch::kCPU) {
      auto getIndicePairFtor =
          functor::CreateSubMIndicePairFunctor<tv::CPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::CPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(gridOut), tv::torch2tv<int>(indicePairs),
          tv::torch2tv<int>(indiceNum), kernelSize32, stride32, padding32,
          dilation32, outSpatialShape32, transpose);
      gridOut.fill_(-1);
    } else {
      auto getIndicePairFtor =
          functor::CreateSubMIndicePairFunctor<tv::GPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::TorchGPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(gridOut), tv::torch2tv<int>(indicePairs),
          tv::torch2tv<int>(indiceNum), kernelSize32, stride32, padding32,
          dilation32, outSpatialShape32, transpose, true);
    }
    return {indices, indicePairs, indiceNum};
  } else {
    torch::Tensor outInds =
        torch::zeros({numAct * kernelVolume, coorDim + 1},
                     torch::dtype(torch::kInt32).device(indices.device()));
    if (indices.device().type() == torch::kCPU) {
      auto getIndicePairFtor =
          functor::CreateConvIndicePairFunctor<tv::CPU, int, int, NDim>();
      numActOut = getIndicePairFtor(
          tv::CPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
          tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
          kernelSize32, stride32, padding32, dilation32, outSpatialShape32,
          transpose, true);
      gridOut.fill_(-1);
    } else {
      auto getIndicePairFtorP1 =
          functor::CreateConvIndicePairFunctorP1<tv::GPU, int, int, NDim>();
      auto getIndicePairFtorP2 =
          functor::CreateConvIndicePairFunctorP2<tv::GPU, int, int, NDim>();
      numActOut = getIndicePairFtorP1(
          tv::TorchGPU(), tv::torch2tv<const int>(indices),
          tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
          tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
          tv::torch2tv<int>(indicePairUnique), kernelSize32, stride32,
          padding32, dilation32, outSpatialShape32, transpose);
      if (numActOut > 0) {
        auto res = torch::_unique(indicePairUnique);
        indicePairUnique = std::get<0>(res);
        numActOut = getIndicePairFtorP2(
            tv::TorchGPU(), tv::torch2tv<const int>(indices),
            tv::torch2tv<int>(outInds), tv::torch2tv<int>(gridOut),
            tv::torch2tv<int>(indicePairs), tv::torch2tv<int>(indiceNum),
            tv::torch2tv<int>(indicePairUnique), outSpatialShape32, transpose,
            true);
      }
    }
    return {outInds.slice(0, 0, numActOut), indicePairs, indiceNum};
  }
}

template <typename T>
torch::Tensor indiceConv(torch::Tensor features, torch::Tensor filters,
                         torch::Tensor indicePairs, torch::Tensor indiceNum,
                         int64_t numActOut, int64_t _inverse, int64_t _subM) {
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

  torch::Tensor output = torch::zeros({numActOut, numOutPlanes}, options);
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

template <typename T>
std::vector<torch::Tensor> indiceConvBackward(torch::Tensor features,
                                              torch::Tensor filters,
                                              torch::Tensor outGrad,
                                              torch::Tensor indicePairs,
                                              torch::Tensor indiceNum,
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
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  auto filterShape = filters.sizes();
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
  torch::Tensor filtersGrad = torch::zeros(filterShape, options);
  torch::Tensor inputBuffer =
      torch::zeros({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::zeros({indicePairMaxSize, numOutPlanes}, options);

  filters = filters.view({-1, numInPlanes, numOutPlanes});
  filtersGrad = filtersGrad.view({-1, numInPlanes, numOutPlanes});
  if (subM) {
    auto filterGradSub = filtersGrad[indicePairMaxOffset];
    torch::mm_out(filterGradSub, features.t(), outGrad);
    torch::mm_out(inputGrad, outGrad, filters[indicePairMaxOffset].t());
  }
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    if (device == torch::kCPU) {
      functor::SparseGatherFunctor<tv::CPU, T, int> gatherFtor;
      functor::SparseGatherFunctor<tv::CPU, T, int> gatherFtorOut;
      gatherFtor(tv::CPU(), tv::torch2tv<T>(inputBuffer),
                 tv::torch2tv<const T>(features),
                 tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                 nHot);
      gatherFtorOut(tv::CPU(), tv::torch2tv<T>(outputBuffer),
                    tv::torch2tv<const T>(outGrad),
                    tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                    nHot);
    } else {
      functor::SparseGatherFunctor<tv::GPU, T, int> gatherFtor;
      functor::SparseGatherFunctor<tv::GPU, T, int> gatherFtorOut;
      gatherFtor(tv::TorchGPU(), tv::torch2tv<T>(inputBuffer),
                 tv::torch2tv<const T>(features),
                 tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                 nHot);
      TV_CHECK_CUDA_ERR();
      gatherFtorOut(tv::TorchGPU(), tv::torch2tv<T>(outputBuffer),
                    tv::torch2tv<const T>(outGrad),
                    tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                    nHot);
      TV_CHECK_CUDA_ERR();
    }
    auto filterGradSub = filtersGrad[i];
    auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr<T>(),
                                             {nHot, numOutPlanes}, options);
    auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr<T>(),
                                            {nHot, numInPlanes}, options);

    torch::mm_out(filterGradSub, inputBufferBlob.t(), outputBufferBlob);
    torch::mm_out(inputBufferBlob, outputBufferBlob, filters[i].t());
    if (device == torch::kCPU) {
      functor::SparseScatterAddFunctor<tv::CPU, T, int> scatterFtor;
      scatterFtor(tv::CPU(), tv::torch2tv<T>(inputGrad),
                  tv::torch2tv<const T>(inputBuffer),
                  tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                  nHot);
    } else {
      functor::SparseScatterAddFunctor<tv::GPU, T, int> scatterFtor;
      scatterFtor(tv::TorchGPU(), tv::torch2tv<T>(inputGrad),
                  tv::torch2tv<const T>(inputBuffer),
                  tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                  nHot);
      TV_CHECK_CUDA_ERR();
    }
  }
  return {inputGrad, filtersGrad.view(filterShape)};
}

template <typename T>
torch::Tensor indiceConvDevelopDontUse(torch::Tensor features,
                                       torch::Tensor filters,
                                       torch::Tensor indicePairs,
                                       torch::Tensor indiceNum,
                                       int64_t numActOut, int64_t _inverse,
                                       int64_t _subM) {
  bool subM = _subM != 0;
  bool inverse = _inverse != 0;

  auto device = features.device().type();
  auto ndim = filters.dim() - 2;
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto numOutPlanes = filters.size(ndim + 1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto totalActsTen = indicePairNumCpu.sum();
  auto totalActs = indicePairNumCpu.data_ptr<int>()[0];
  auto indicePairMaxSizeIter =
      std::max_element(indicePairNumCpu.data_ptr<int>(),
                       indicePairNumCpu.data_ptr<int>() + kernelVolume);
  int indicePairMaxOffset =
      indicePairMaxSizeIter - indicePairNumCpu.data_ptr<int>();
  int indicePairMaxSize = *indicePairMaxSizeIter;
  std::vector<int> indicePairNumVec(
      indicePairNumCpu.data_ptr<int>(),
      indicePairNumCpu.data_ptr<int>() + kernelVolume);
  indicePairNumVec.erase(indicePairNumVec.begin() + indicePairMaxOffset);
  int subRuleMaxSize =
      *std::max_element(indicePairNumVec.begin(), indicePairNumVec.end());
  if (subM) {
    indicePairMaxSize = subRuleMaxSize;
  }
  auto timer = spconv::CudaContextTimer<>();
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  // auto indicePairOptions =
  //     torch::TensorOptions().dtype(torch::kInt64).device(indicePairs.device());

  torch::Tensor output = torch::zeros({numActOut, numOutPlanes}, options);
  torch::Tensor inputBuffer =
      torch::zeros({kernelVolume, indicePairMaxSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::zeros({kernelVolume, indicePairMaxSize, numOutPlanes}, options);
  filters = filters.view({-1, numInPlanes, numOutPlanes});
  std::cout << "create time " << timer.report() / 1000.0 << std::endl;
  if (subM) {  // the center index of subm conv don't need gather and scatter
               // add.
    torch::mm_out(output, features, filters[indicePairMaxOffset]);
  }
  double totalGatherTime = 0;
  double totalGEMMTime = 0;
  double totalSAddTime = 0;
  // auto timer = spconv::CudaContextTimer<>();
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    //
    auto outputBufferBlob = torch::from_blob(outputBuffer[i].data_ptr<T>(),
                                             {nHot, numOutPlanes}, options);
    auto inputBufferBlob = torch::from_blob(inputBuffer[i].data_ptr<T>(),
                                            {nHot, numInPlanes}, options);
    if (device == torch::kCPU) {
      functor::SparseGatherFunctor<tv::CPU, T, int> gatherFtor;
      gatherFtor(tv::CPU(), tv::torch2tv<T>(inputBufferBlob),
                 tv::torch2tv<const T>(features),
                 tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                 nHot);
    } else {
      functor::SparseGatherFunctor<tv::GPU, T, int> gatherFtor;
      gatherFtor(tv::TorchGPU(), tv::torch2tv<T>(inputBufferBlob),
                 tv::torch2tv<const T>(features),
                 tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                 nHot);
      TV_CHECK_CUDA_ERR();
    }
    // }
    // for (int i = 0; i < kernelVolume; ++i) {
    // totalGatherTime += timer.report() / 1000.0;
    // auto outputBufferBlob = torch::from_blob(outputBuffer[i].data_ptr<T>(),
    // {nHot, numOutPlanes}, options);
  }
  // totalGatherTime += timer.report() / 1000.0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    auto outputBufferBlob = torch::from_blob(outputBuffer[i].data_ptr<T>(),
                                             {nHot, numOutPlanes}, options);
    auto inputBufferBlob = torch::from_blob(inputBuffer[i].data_ptr<T>(),
                                            {nHot, numInPlanes}, options);

    torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);
  }
  // totalGEMMTime += timer.report() / 1000.0;
  // totalGEMMTime += timer.report() / 1000.0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    auto outputBufferBlob = torch::from_blob(outputBuffer[i].data_ptr<T>(),
                                             {nHot, numOutPlanes}, options);
    auto inputBufferBlob = torch::from_blob(inputBuffer[i].data_ptr<T>(),
                                            {nHot, numInPlanes}, options);

    if (device == torch::kCPU) {
      functor::SparseScatterAddFunctor<tv::CPU, T, int> scatterFtor;
      scatterFtor(tv::CPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(outputBufferBlob),
                  tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                  nHot, true);
    } else {
      functor::SparseScatterAddFunctor<tv::GPU, T, int> scatterFtor;
      scatterFtor(tv::TorchGPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(outputBufferBlob),
                  tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                  nHot, true);
      TV_CHECK_CUDA_ERR();
    }
    // totalSAddTime += timer.report() / 1000.0;
  }
  // totalSAddTime += timer.report() / 1000.0;
  // std::cout << "gather time " << totalGatherTime << std::endl;
  // std::cout << "gemm time " << totalGEMMTime << std::endl;
  // std::cout << "scatteradd time " << totalSAddTime << std::endl;
  return output;
}

}  // namespace spconv

#endif
