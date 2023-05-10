# Inference

## Introduction

We provide scripts for multi-modality/single-modality (LiDAR-based/vision-based), indoor/outdoor 3D detection and 3D semantic segmentation demos. The pre-trained models can be downloaded from [model zoo](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docs/en/model_zoo.md). We provide pre-processed sample data from KITTI, SUN RGB-D, nuScenes and ScanNet dataset. You can use any other data following our pre-processing steps.

## Testing

### 3D Detection

#### Point cloud demo

To test a 3D detector on point cloud data, simply run:

```shell
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]
```

The visualization results including a point cloud and predicted 3D bounding boxes will be saved in `${OUT_DIR}/PCD_NAME`, which you can open using [MeshLab](http://www.meshlab.net/). Note that if you set the flag `--show`, the prediction result will be displayed online using [Open3D](http://www.open3d.org/).

Example on KITTI data using [PointPillars model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth):

```shell
python demo/pcd_demo.py demo/data/kitti/000008.bin configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py ${CHECKPOINT_FILE} --show
```

Example on SUN RGB-D data using [VoteNet model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/votenet/votenet_16x8_sunrgbd-3d-10class/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth):

```shell
python demo/pcd_demo.py demo/data/sunrgbd/sunrgbd_000017.bin configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT_FILE} --show
```

#### Monocular 3D demo

To test a monocular 3D detector on image data, simply run:

```shell
python demo/mono_det_demo.py ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--cam-type ${CAM_TYPE}] [--score-thr ${SCORE-THR}] [--out-dir ${OUT_DIR}] [--show]
```

where the `ANNOTATION_FILE` should provide the 3D to 2D projection matrix (camera intrinsic matrix), and `CAM_TYPE` should be specified according to dataset. For example, if you want to inference on the front camera image, the `CAM_TYPE` should be set as `CAM_2` for KITTI, and `CAM_FRONT` for nuScenes. By specifying `CAM_TYPE`, you can even infer on any camera images for datasets with multi-view cameras, such as nuScenes and Waymo. `SCORE-THR` is the 3D bbox threshold while visualization. The visualization results including an image and its predicted 3D bounding boxes projected on the image will be saved in `${OUT_DIR}/IMG_NAME`.

Example on KITTI data using [PGD model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth):

```shell
python demo/mono_det_demo.py demo/data/kitti/000008.png demo/data/kitti/000008.pkl  configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py ${CHECKPOINT_FILE}  --show --cam-type CAM2 --score-thr 8
```

**Note**: For PGD, the prediction score is not among (0, 1).

Example on nuScenes data using [FCOS3D model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth):

```shell
python demo/mono_det_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl  configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py ${CHECKPOINT_FILE}  --show --cam-type CAM_BACK
```

**Note** that when visualizing results of monocular 3D detection for flipped images, the camera intrinsic matrix should also be modified accordingly. See more details and examples in PR [#744](https://github.com/open-mmlab/mmdetection3d/pull/744).

#### Multi-modality demo

To test a 3D detector on multi-modality data (typically point cloud and image), simply run:

```shell
python demo/multi_modality_demo.py ${PCD_FILE} ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]
```

where the `ANNOTATION_FILE` should provide the 3D to 2D projection matrix. The visualization results including a point cloud, an image, predicted 3D bounding boxes and their projection on the image will be saved in `${OUT_DIR}/PCD_NAME`.

Example on KITTI data using [MVX-Net model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth):

```shell
python demo/multi_modality_demo.py demo/data/kitti/000008.bin demo/data/kitti/000008.png demo/data/kitti/000008.pkl configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py ${CHECKPOINT_FILE} --cam-type CAM2 --show
```

Example on SUN RGB-D data using [ImVoteNet model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210819_192851-1bcd1b97.pth):

```shell
python demo/multi_modality_demo.py demo/data/sunrgbd/000017.bin demo/data/sunrgbd/000017.jpg demo/data/sunrgbd/sunrgbd_000017_infos.pkl configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${CHECKPOINT_FILE} --cam-type CAM0 --show --score-thr 0.6
```

Example on NuScenes data using [BEVFusion model](https://drive.google.com/file/d/1QkvbYDk4G2d6SZoeJqish13qSyXA4lp3/view?usp=share_link):

```shell
python demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl projects/BEVFusion/configs/bevfusion_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ${CHECKPOINT_FILE} --cam-type all --score-thr 0.2 --show
```

### 3D Segmentation

To test a 3D segmentor on point cloud data, simply run:

```shell
python demo/pcd_seg_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--out-dir ${OUT_DIR}] [--show]
```

The visualization results including a point cloud and its predicted 3D segmentation mask will be saved in `${OUT_DIR}/PCD_NAME`.

Example on ScanNet data using [PointNet++ (SSG) model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class_20210514_143644-ee73704a.pth):

```shell
python demo/pcd_seg_demo.py demo/data/scannet/scene0000_00.bin configs/pointnet2/pointnet2_ssg_2xb16-cosine-200e_scannet-seg.py ${CHECKPOINT_FILE} --show
```
