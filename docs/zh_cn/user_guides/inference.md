# 推理

## 介绍

我们提供了多模态/单模态（基于激光雷达/图像）、室内/室外场景的 3D 检测和 3D 语义分割样例的脚本，预训练模型可以从 [Model Zoo](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docs/zh_cn/model_zoo.md) 下载。我们也提供了 KITTI、SUN RGB-D、nuScenes 和 ScanNet 数据集的预处理样本数据，你可以根据我们的预处理步骤使用任何其它数据。

## 测试

### 3D 检测

#### 点云样例

在点云数据上测试 3D 检测器，运行：

```shell
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]
```

点云和预测 3D 框的可视化结果会被保存在 `${OUT_DIR}/PCD_NAME`，它可以使用 [MeshLab](http://www.meshlab.net/) 打开。注意如果你设置了 `--show`，通过 [Open3D](http://www.open3d.org/) 可以在线显示预测结果。

在 KITTI 数据上测试 [PointPillars 模型](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth)：

```shell
python demo/pcd_demo.py demo/data/kitti/000008.bin configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py ${CHECKPOINT_FILE} --show
```

在 SUN RGB-D 数据上测试 [VoteNet 模型](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/votenet/votenet_16x8_sunrgbd-3d-10class/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth)：

```shell
python demo/pcd_demo.py demo/data/sunrgbd/sunrgbd_000017.bin configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT_FILE} --show
```

#### 单目 3D 样例

在图像数据上测试单目 3D 检测器，运行：

```shell
python demo/mono_det_demo.py ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--out-dir ${OUT_DIR}] [--show]
```

`ANNOTATION_FILE` 需要提供 3D 到 2D 的仿射矩阵（相机内参矩阵），可视化结果会被保存在 `${OUT_DIR}/PCD_NAME`，其中包括图像以及预测 3D 框在图像上的投影。

在 KITTI 数据上测试 [PGD 模型](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth)：

```shell
python demo/mono_det_demo.py demo/data/kitti/000008.png demo/data/kitti/000008.pkl  configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py ${CHECKPOINT_FILE}  --show --cam-type CAM2 --score-thr 8
```

**注意**： PGD 方法的预测框分数并不是在 (0, 1) 之间

在 nuScenes 数据上测试 [FCOS3D 模型](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth)：

```shell
python demo/mono_det_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl  configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py ${CHECKPOINT_FILE}  --show --cam-type CAM_BACK
```

**注意**： 当对翻转图像可视化单目 3D 检测结果是，相机内参矩阵也应该相应修改。在 PR [#744](https://github.com/open-mmlab/mmdetection3d/pull/744) 中可以了解更多细节和示例。

#### 多模态样例

在多模态数据（通常是点云和图像）上测试 3D 检测器，运行：

```shell
python demo/multi_modality_demo.py ${PCD_FILE} ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]
```

`ANNOTATION_FILE` 需要提供 3D 到 2D 的仿射矩阵，可视化结果会被保存在 `${OUT_DIR}/PCD_NAME`，其中包括点云、图像、预测的 3D 框以及它们在图像上的投影。

在 KITTI 数据上测试 [MVX-Net 模型](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth)：

```shell
python demo/multi_modality_demo.py demo/data/kitti/000008.bin demo/data/kitti/000008.png demo/data/kitti/000008.pkl configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py ${CHECKPOINT_FILE} --cam-type CAM2 --show
```

在 SUN RGB-D 数据上测试 [ImVoteNet 模型](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210819_192851-1bcd1b97.pth)：

```shell
python demo/multi_modality_demo.py demo/data/sunrgbd/000017.bin demo/data/sunrgbd/000017.jpg demo/data/sunrgbd/sunrgbd_000017_infos.pkl configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${CHECKPOINT_FILE} --cam-type CAM0 --show --score-thr 0.6
```

在 NuScenes 数据上测试 [BEVFusion 模型](https://drive.google.com/file/d/1QkvbYDk4G2d6SZoeJqish13qSyXA4lp3/view?usp=share_link)

```shell
python demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl projects/BEVFusion/configs/bevfusion_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ${CHECKPOINT_FILE} --cam-type all --score-thr 0.2 --show
```

### 3D 分割

在点云数据上测试 3D 分割器，运行：

```shell
python demo/pc_seg_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--out-dir ${OUT_DIR}] [--show]
```

可视化结果会被保存在 `${OUT_DIR}/PCD_NAME`，其中包括点云以及预测的 3D 分割掩码。

在 ScanNet 数据上测试 [PointNet++ (SSG) 模型](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class_20210514_143644-ee73704a.pth)：

```shell
python demo/pcd_seg_demo.py demo/data/scannet/scene0000_00.bin configs/pointnet2/pointnet2_ssg_2xb16-cosine-200e_scannet-seg.py ${CHECKPOINT_FILE} --show
```
