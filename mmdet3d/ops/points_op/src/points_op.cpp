#include <pybind11/pybind11.h>
// must include pybind11/eigen.h if using eigen matrix as arguments.
// must include pybind11/stl.h if using containers in STL in arguments.
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <vector>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
namespace py = pybind11;
using namespace pybind11::literals;

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")

template <typename DType, int NDim>
int points_to_bev_kernel(py::array_t<DType> points,
                         std::vector<DType> voxel_size,
                         std::vector<DType> coors_range,
                         py::array_t<DType> bev,
                         int scale)
{
  auto points_rw = points.template mutable_unchecked<2>();
  auto N = points_rw.shape(0);
  auto bev_rw = bev.template mutable_unchecked<NDim>();

  int zdim_minus_1 = bev.shape(0)-1;
  int zdim_minus_2 = bev.shape(0)-2;

  constexpr int ndim_minus_1 = NDim - 1;
  bool failed = false;
  int coor[NDim];
  int c;
  int grid_size[NDim];
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  DType intensity;
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed)
      continue;
    bev_rw(coor[0], coor[1], coor[2])=1;
    bev_rw(zdim_minus_1, coor[1], coor[2])+=1;
    intensity = points_rw(i, 3);
    if (intensity > bev_rw(zdim_minus_2, coor[1], coor[2]))
        bev_rw(zdim_minus_2, coor[1], coor[2])=intensity;

  }
  return 1;
}

template <typename DType>
py::array_t<bool> points_bound_kernel(py::array_t<DType> points,
                        std::vector<DType> lower_bound,
                        std::vector<DType> upper_bound
                        ){

    auto points_ptr = points.template mutable_unchecked<2>();


    int N = points_ptr.shape(0);
    int ndim = points_ptr.shape(1);
    auto keep = py::array_t<bool>(N);

    auto keep_ptr = keep.mutable_unchecked<1>();

    bool success = 0;
    for (int i = 0; i < N; i++){
        success = 1;
        for (int j=0; j<(ndim-1); j++){
            if(points_ptr(i, j) < lower_bound[j] || points_ptr(i, j) >= upper_bound[j]){
                success = 0;
                break;
            }
        }
        keep_ptr(i) = success;
    }
    return keep;
}

int pt_in_box3d_cpu(float x, float y, float z, float cx, float cy, float bottom_z, float w, float l, float h, float angle){
    float max_dis = 10.0, x_rot, y_rot, cosa, sina, cz;
    int in_flag;
    cz = bottom_z + h / 2.0;
    if ((fabsf(x - cx) > max_dis) || (fabsf(z - cz) > h / 2.0) || (fabsf(y - cy) > max_dis)){
        return 0;
    }
    cosa = cos(angle); sina = sin(angle);
    x_rot = (x - cx) * cosa + (y - cy) * (-sina);
    y_rot = (x - cx) * sina + (y - cy) * cosa;

    in_flag = (x_rot >= -w / 2.0) & (x_rot <= w / 2.0) & (y_rot >= -l / 2.0) & (y_rot <= l / 2.0);
    return in_flag;
}

int pts_in_boxes3d_cpu(at::Tensor pts, at::Tensor boxes3d, at::Tensor pts_flag, at::Tensor reg_target){
    // param pts: (N, 3)
    // param boxes3d: (M, 7)  [x, y, z, h, w, l, ry]
    // param pts_flag: (M, N)
    // param reg_target: (N, 3), center offsets

    CHECK_CONTIGUOUS(pts_flag);
    CHECK_CONTIGUOUS(pts);
    CHECK_CONTIGUOUS(boxes3d);
    CHECK_CONTIGUOUS(reg_target);

    long boxes_num = boxes3d.size(0);
    long pts_num = pts.size(0);

    int * pts_flag_flat = pts_flag.data<int>();
    float * pts_flat = pts.data<float>();
    float * boxes3d_flat = boxes3d.data<float>();
    float * reg_target_flat = reg_target.data<float>();

//    memset(assign_idx_flat, -1, boxes_num * pts_num * sizeof(int));
//    memset(reg_target_flat, 0, pts_num * sizeof(float));

    int i, j, cur_in_flag;
    for (i = 0; i < boxes_num; i++){
        for (j = 0; j < pts_num; j++){
            cur_in_flag = pt_in_box3d_cpu(pts_flat[j * 3], pts_flat[j * 3 + 1], pts_flat[j * 3 + 2], boxes3d_flat[i * 7],
                                          boxes3d_flat[i * 7 + 1], boxes3d_flat[i * 7 + 2], boxes3d_flat[i * 7 + 3],
                                          boxes3d_flat[i * 7 + 4], boxes3d_flat[i * 7 + 5], boxes3d_flat[i * 7 + 6]);
            pts_flag_flat[i * pts_num + j] = cur_in_flag;
            if(cur_in_flag==1){
                reg_target_flat[j*3] = pts_flat[j*3] - boxes3d_flat[i*7];
                reg_target_flat[j*3+1] = pts_flat[j*3+1] - boxes3d_flat[i*7+1];
                reg_target_flat[j*3+2] = pts_flat[j*3+2] - (boxes3d_flat[i*7+2] + boxes3d_flat[i*7+3] / 2.0);
            }
        }
    }
    return 1;
}

template <typename DType>
py::array_t<bool> points_in_bbox3d_np(py::array_t<DType> points,
                        py::array_t<DType> boxes3d)
{

    //const DType* points_ptr = static_cast<DType*>(points.request().ptr);
    //const DType* boxes3d_ptr = static_cast<DType*>(boxes3d.request().ptr);
    //const DType* boxes3d_ptr = boxes3d.data();

    auto points_ptr = points.template mutable_unchecked<2>();
    auto boxes3d_ptr = boxes3d.template mutable_unchecked<2>();

    int N = points.shape(0);
    int M = boxes3d.shape(0);

    auto keep = py::array_t<bool>({N,M});

    //int * keep_ptr = keep.mutable_data();
    //int * keep_ptr = static_cast<int*>(keep.request().ptr);
    auto keep_ptr = keep.mutable_unchecked<2>();

    int i, j, cur_in_flag;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            cur_in_flag = pt_in_box3d_cpu(
                points_ptr(j, 0), points_ptr(j, 1), points_ptr(j, 2),
                boxes3d_ptr(i, 0), boxes3d_ptr(i, 1), boxes3d_ptr(i, 2),
                boxes3d_ptr(i, 3), boxes3d_ptr(i, 4), boxes3d_ptr(i, 5),
                boxes3d_ptr(i, 6)
            );

            keep_ptr(j, i) = (bool) cur_in_flag;
        }
    }

    return keep;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "pybind11 example plugin";      // module doc string
  m.def("points_to_bev_kernel",                              // function name
        &points_to_bev_kernel<float,3>,                               // function pointer
        "function of converting points to voxel" //function doc string
       );
  m.def("points_bound_kernel",                              // function name
        &points_bound_kernel<float>,                               // function pointer
        "function of filtering points" //function doc string
       );
  m.def("pts_in_boxes3d", &pts_in_boxes3d_cpu, "points in boxes3d (CPU)");
  m.def("points_in_bbox3d_np", &points_in_bbox3d_np<float>, "points in boxes3d using numpy (CPU)");
}