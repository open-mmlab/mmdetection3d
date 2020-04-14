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
#include <algorithm>
#include <iostream>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tensorview/tensorview.h>

namespace py = pybind11;

template <typename T, typename TPyObject>
std::vector<T> array2Vector(TPyObject arr){
    py::array arr_np = arr;
    size_t size = arr.attr("size").template cast<size_t>();
    py::array_t<T> arr_cc = arr_np;
    std::vector<T> data(arr_cc.data(), arr_cc.data() + size);
    return data;
}

template <typename T>
std::vector<T> arrayT2Vector(py::array_t<T> arr)
{
  std::vector<T> data(arr.data(), arr.data() + arr.size());
  return data;
}

template <typename T, typename TPyObject>
tv::TensorView<T> array2TensorView(TPyObject arr){
    py::array arr_np = arr;
    py::array_t<T> arr_cc = arr_np;
    tv::Shape shape;
    for (int i = 0; i < arr_cc.ndim(); ++i){
        shape.push_back(arr_cc.shape(i));
    }
    return tv::TensorView<T>(arr_cc.mutable_data(), shape);
}
template <typename T>
tv::TensorView<T> arrayT2TensorView(py::array_t<T> arr){
    tv::Shape shape;
    for (int i = 0; i < arr.ndim(); ++i){
        shape.push_back(arr.shape(i));
    }
    return tv::TensorView<T>(arr.mutable_data(), shape);
}
