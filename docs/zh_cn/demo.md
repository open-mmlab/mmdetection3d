# 样例

# 介绍

我们提供了多模态/单模态（基于激光雷达/图像）、室内/室外场景的 3D 检测和 3D 语义分割样例的脚本，预训练模型可以从 [Model Zoo](https://github.com/open-mmlab/mmdetection3d/blob/master/docs_zh-CN/model_zoo.md/) 下载。我们也提供了 KITTI、SUN RGB-D、nuScenes 和 ScanNet 数据集的预处理样本数据，你可以根据我们的预处理步骤使用任何其它数据。

## 测试

### 3D 检测

#### 单模态样例

在点云数据上测试 3D 检测器，运行：

```shell
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]
```

点云和预测 3D 框的可视化结果会被保存在 `${OUT_DIR}/PCD_NAME`，它可以使用 [MeshLab](http://www.meshlab.net/) 打开。注意如果你设置了 `--show`，通过 [Open3D](http://www.open3d.org/) 可以在线显示预测结果。

在 KITTI 数据上测试 [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) 模型：

```shell
python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth
```

在 SUN RGB-D 数据上测试 [VoteNet](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/votenet) 模型：

```shell
python demo/pcd_demo.py demo/data/sunrgbd/sunrgbd_000017.bin configs/votenet/votenet_16x8_sunrgbd-3d-10class.py checkpoints/votenet_16x8_sunrgbd-3d-10class_20200620_230238-4483c0c0.pth
```

如果你正在使用的 mmdetection3d 版本 >= 0.6.0，记住转换 VoteNet 的模型权重文件，查看 [README](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/votenet/README.md/) 来获取转换模型权重文件的详细说明。

#### 多模态样例

在多模态数据（通常是点云和图像）上测试 3D 检测器，运行：

```shell
python demo/multi_modality_demo.py ${PCD_FILE} ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]
```

`ANNOTATION_FILE` 需要提供 3D 到 2D 的仿射矩阵，可视化结果会被保存在 `${OUT_DIR}/PCD_NAME`，其中包括点云、图像、预测的 3D 框以及它们在图像上的投影。

在 KITTI 数据上测试 [MVX-Net](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/mvxnet) 模型：

```shell
python demo/multi_modality_demo.py demo/data/kitti/kitti_000008.bin demo/data/kitti/kitti_000008.png demo/data/kitti/kitti_000008_infos.pkl configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth
```

在 SUN RGB-D 数据上测试 [ImVoteNet](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/imvotenet) 模型：

```shell
python demo/multi_modality_demo.py demo/data/sunrgbd/sunrgbd_000017.bin demo/data/sunrgbd/sunrgbd_000017.jpg demo/data/sunrgbd/sunrgbd_000017_infos.pkl configs/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class.py checkpoints/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210323_184021-d44dcb66.pth
```

### 单目 3D 检测

在图像数据上测试单目 3D 检测器，运行：

```shell
python demo/mono_det_demo.py ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--out-dir ${OUT_DIR}] [--show]
```

`ANNOTATION_FILE` 需要提供 3D 到 2D 的仿射矩阵（相机内参矩阵），可视化结果会被保存在 `${OUT_DIR}/PCD_NAME`，其中包括图像以及预测 3D 框在图像上的投影。

在 nuScenes 数据上测试 [FCOS3D](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/fcos3d) 模型：

```shell
python demo/mono_det_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525_mono3d.coco.json configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
```

### 3D 分割

在点云数据上测试 3D 分割器，运行：

```shell
python demo/pc_seg_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--out-dir ${OUT_DIR}] [--show]
```

可视化结果会被保存在 `${OUT_DIR}/PCD_NAME`，其中包括点云以及预测的 3D 分割掩码。

在 ScanNet 数据上测试 [PointNet++ (SSG)](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointnet2) 模型：

```shell
python demo/pc_seg_demo.py demo/data/scannet/scene0000_00.bin configs/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py checkpoints/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class_20210514_143644-ee73704a.pth
```
