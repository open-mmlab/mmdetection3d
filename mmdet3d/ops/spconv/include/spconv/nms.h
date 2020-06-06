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

#ifndef NMS_CPU_H
#define NMS_CPU_H
#include <pybind11/pybind11.h>
// must include pybind11/stl.h if using containers in STL in arguments.
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <boost/geometry.hpp>
#include <vector>

#include "box_iou.h"
#include "nms_gpu.h"
namespace spconv {
namespace py = pybind11;
using namespace pybind11::literals;

template <typename DType>
std::vector<int> non_max_suppression_cpu(py::array_t<DType> boxes,
                                         py::array_t<int> order, DType thresh,
                                         DType eps = 0) {
  auto ndets = boxes.shape(0);
  auto boxes_r = boxes.template unchecked<2>();
  auto order_r = order.template unchecked<1>();
  auto suppressed = zeros<int>({int(ndets)});
  auto suppressed_rw = suppressed.template mutable_unchecked<1>();
  auto area = zeros<DType>({int(ndets)});
  auto area_rw = area.template mutable_unchecked<1>();
  // get areas
  for (int i = 0; i < ndets; ++i) {
    area_rw(i) = (boxes_r(i, 2) - boxes_r(i, 0) + eps) *
                 (boxes_r(i, 3) - boxes_r(i, 1) + eps);
  }
  std::vector<int> keep;
  int i, j;
  DType xx1, xx2, w, h, inter, ovr;
  for (int _i = 0; _i < ndets; ++_i) {
    i = order_r(_i);
    if (suppressed_rw(i) == 1) continue;
    keep.push_back(i);
    for (int _j = _i + 1; _j < ndets; ++_j) {
      j = order_r(_j);
      if (suppressed_rw(j) == 1) continue;
      xx2 = std::min(boxes_r(i, 2), boxes_r(j, 2));
      xx1 = std::max(boxes_r(i, 0), boxes_r(j, 0));
      w = xx2 - xx1 + eps;
      if (w > 0) {
        xx2 = std::min(boxes_r(i, 3), boxes_r(j, 3));
        xx1 = std::max(boxes_r(i, 1), boxes_r(j, 1));
        h = xx2 - xx1 + eps;
        if (h > 0) {
          inter = w * h;
          ovr = inter / (area_rw(i) + area_rw(j) - inter);
          if (ovr >= thresh) suppressed_rw(j) = 1;
        }
      }
    }
  }
  return keep;
}

template <typename DType>
std::vector<int> rotate_non_max_suppression_cpu(py::array_t<DType> box_corners,
                                                py::array_t<int> order,
                                                py::array_t<DType> standup_iou,
                                                DType thresh) {
  auto ndets = box_corners.shape(0);
  auto box_corners_r = box_corners.template unchecked<3>();
  auto order_r = order.template unchecked<1>();
  auto suppressed = zeros<int>({int(ndets)});
  auto suppressed_rw = suppressed.template mutable_unchecked<1>();
  auto standup_iou_r = standup_iou.template unchecked<2>();
  std::vector<int> keep;
  int i, j;

  namespace bg = boost::geometry;
  typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
  typedef bg::model::polygon<point_t> polygon_t;
  polygon_t poly, qpoly;
  std::vector<polygon_t> poly_inter, poly_union;
  DType inter_area, union_area, overlap;

  for (int _i = 0; _i < ndets; ++_i) {
    i = order_r(_i);
    if (suppressed_rw(i) == 1) continue;
    keep.push_back(i);
    for (int _j = _i + 1; _j < ndets; ++_j) {
      j = order_r(_j);
      if (suppressed_rw(j) == 1) continue;
      if (standup_iou_r(i, j) <= 0.0) continue;
      // std::cout << "pre_poly" << std::endl;
      try {
        bg::append(poly,
                   point_t(box_corners_r(i, 0, 0), box_corners_r(i, 0, 1)));
        bg::append(poly,
                   point_t(box_corners_r(i, 1, 0), box_corners_r(i, 1, 1)));
        bg::append(poly,
                   point_t(box_corners_r(i, 2, 0), box_corners_r(i, 2, 1)));
        bg::append(poly,
                   point_t(box_corners_r(i, 3, 0), box_corners_r(i, 3, 1)));
        bg::append(poly,
                   point_t(box_corners_r(i, 0, 0), box_corners_r(i, 0, 1)));
        bg::append(qpoly,
                   point_t(box_corners_r(j, 0, 0), box_corners_r(j, 0, 1)));
        bg::append(qpoly,
                   point_t(box_corners_r(j, 1, 0), box_corners_r(j, 1, 1)));
        bg::append(qpoly,
                   point_t(box_corners_r(j, 2, 0), box_corners_r(j, 2, 1)));
        bg::append(qpoly,
                   point_t(box_corners_r(j, 3, 0), box_corners_r(j, 3, 1)));
        bg::append(qpoly,
                   point_t(box_corners_r(j, 0, 0), box_corners_r(j, 0, 1)));
        bg::intersection(poly, qpoly, poly_inter);
      } catch (const std::exception &e) {
        std::cout << "box i corners:" << std::endl;
        for (int k = 0; k < 4; ++k) {
          std::cout << box_corners_r(i, k, 0) << " " << box_corners_r(i, k, 1)
                    << std::endl;
        }
        std::cout << "box j corners:" << std::endl;
        for (int k = 0; k < 4; ++k) {
          std::cout << box_corners_r(j, k, 0) << " " << box_corners_r(j, k, 1)
                    << std::endl;
        }
        // throw e;
        continue;
      }
      // std::cout << "post_poly" << std::endl;
      // std::cout << "post_intsec" << std::endl;
      if (!poly_inter.empty()) {
        inter_area = bg::area(poly_inter.front());
        // std::cout << "pre_union" << " " << inter_area << std::endl;
        bg::union_(poly, qpoly, poly_union);
        /*
        if (poly_union.empty()){
            std::cout << "intsec area:" << " " << inter_area << std::endl;
            std::cout << "box i corners:" << std::endl;
            for(int k = 0; k < 4; ++k){
                std::cout << box_corners_r(i, k, 0) << " " << box_corners_r(i,
        k, 1) << std::endl;
            }
            std::cout << "box j corners:" <<  std::endl;
            for(int k = 0; k < 4; ++k){
                std::cout << box_corners_r(j, k, 0) << " " << box_corners_r(j,
        k, 1) << std::endl;
            }
        }*/
        // std::cout << "post_union" << poly_union.empty() << std::endl;
        if (!poly_union.empty()) {  // ignore invalid box
          union_area = bg::area(poly_union.front());
          // std::cout << "post union area" << std::endl;
          // std::cout << union_area << "debug" << std::endl;
          overlap = inter_area / union_area;
          if (overlap >= thresh) suppressed_rw(j) = 1;
          poly_union.clear();
        }
      }
      poly.clear();
      qpoly.clear();
      poly_inter.clear();
    }
  }
  return keep;
}

constexpr int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename DType>
int non_max_suppression(py::array_t<DType> boxes, py::array_t<int> keep_out,
                        DType nms_overlap_thresh, int device_id) {
  py::buffer_info info = boxes.request();
  auto boxes_ptr = static_cast<DType *>(info.ptr);
  py::buffer_info info_k = keep_out.request();
  auto keep_out_ptr = static_cast<int *>(info_k.ptr);

  return _nms_gpu<DType, threadsPerBlock>(keep_out_ptr, boxes_ptr,
                                          boxes.shape(0), boxes.shape(1),
                                          nms_overlap_thresh, device_id);
}

}  // namespace spconv
#endif
