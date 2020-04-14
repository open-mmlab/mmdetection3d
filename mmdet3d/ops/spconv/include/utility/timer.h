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

#pragma once
#include <chrono>
#include <cuda_runtime_api.h>
#include <iostream>

namespace spconv {

template <typename TimeT = std::chrono::microseconds> struct CudaContextTimer {
  CudaContextTimer() {
    cudaDeviceSynchronize();
    mCurTime = std::chrono::steady_clock::now();
  }
  typename TimeT::rep report() {
    cudaDeviceSynchronize();
    auto duration = std::chrono::duration_cast<TimeT>(
        std::chrono::steady_clock::now() - mCurTime);
    auto res = duration.count();
    mCurTime = std::chrono::steady_clock::now();
    return res;
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> mCurTime;
};

template <typename TimeT = std::chrono::microseconds> struct CPUTimer {
  CPUTimer() { mCurTime = std::chrono::steady_clock::now(); }
  typename TimeT::rep report() {
    auto duration = std::chrono::duration_cast<TimeT>(
        std::chrono::steady_clock::now() - mCurTime);
    auto res = duration.count();
    mCurTime = std::chrono::steady_clock::now();
    return res;
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> mCurTime;
};

} // namespace spconv
