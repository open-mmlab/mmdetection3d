# 模型库

## 通用设置

- 使用分布式训练；
- 为了和其他代码库做公平对比，本文展示的是使用 `torch.cuda.max_memory_allocated()` 在 8 个 GPUs 上得到的最大 GPU 显存占用值，需要注意的是，这些显存占用值通常小于 `nvidia-smi` 显示出来的显存占用值；
- 在模型库中所展示的推理时间是包括网络前向传播和后处理所需的总时间，不包括数据加载所需的时间，模型库中所展示的结果均由 [benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/benchmark.py) 脚本文件在 2000 张图像上所计算的平均时间。

## 基准结果

### SECOND

请参考 [SECOND](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/second) 获取更多的细节，我们在 KITTI 和 Waymo 数据集上都给出了相应的基准结果。

### PointPillars

请参考 [PointPillars](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars) 获取更多细节，我们在 KITTI 、nuScenes 、Lyft 、Waymo 数据集上给出了相应的基准结果。

### Part-A2

请参考 [Part-A2](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/parta2) 获取更多细节。

### VoteNet

请参考 [VoteNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/votenet) 获取更多细节，我们在 ScanNet 和 SUNRGBD 数据集上给出了相应的基准结果。

### Dynamic Voxelization

请参考 [Dynamic Voxelization](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/dynamic_voxelization) 获取更多细节。

### MVXNet

请参考 [MVXNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/mvxnet) 获取更多细节。

### RegNetX

请参考 [RegNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/regnet) 获取更多细节，我们将 pointpillars 的主干网络替换成 RegNetX，并在 nuScenes 和 Lyft 数据集上给出了相应的基准结果。

### nuImages

我们在 [nuImages 数据集](https://www.nuscenes.org/nuimages) 上也提供基准模型，请参考 [nuImages](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/nuimages) 获取更多细节，我们在该数据集上提供 Mask R-CNN ， Cascade Mask R-CNN 和 HTC 的结果。

### H3DNet

请参考 [H3DNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/h3dnet) 获取更多细节。

### 3DSSD

请参考 [3DSSD](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/3dssd) 获取更多细节。

### CenterPoint

请参考 [CenterPoint](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/centerpoint) 获取更多细节。

### SSN

请参考 [SSN](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/ssn) 获取更多细节，我们将 pointpillars 中的检测头替换成 SSN 模型中所使用的 ‘shape-aware grouping heads’，并在 nuScenes 和 Lyft 数据集上给出了相应的基准结果。

### ImVoteNet

请参考 [ImVoteNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/imvotenet) 获取更多细节，我们在 SUNRGBD 数据集上给出了相应的结果。

### FCOS3D

请参考 [FCOS3D](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/fcos3d) 获取更多细节，我们在 nuScenes 数据集上给出了相应的结果。

### PointNet++

请参考 [PointNet++](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointnet2) 获取更多细节，我们在 ScanNet 和 S3DIS 数据集上给出了相应的结果。

### Group-Free-3D

请参考 [Group-Free-3D](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/groupfree3d) 获取更多细节，我们在 ScanNet 数据集上给出了相应的结果。

### ImVoxelNet

请参考 [ImVoxelNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/imvoxelnet) 获取更多细节，我们在 KITTI 数据集上给出了相应的结果。

### PAConv

请参考 [PAConv](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/paconv) 获取更多细节，我们在 S3DIS 数据集上给出了相应的结果.

### DGCNN

请参考 [DGCNN](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/dgcnn) 获取更多细节，我们在 S3DIS 数据集上给出了相应的结果.

### SMOKE

请参考 [SMOKE](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/smoke) 获取更多细节，我们在 KITTI 数据集上给出了相应的结果.

### PGD

请参考 [PGD](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/pgd) 获取更多细节，我们在 KITTI 和 nuScenes 数据集上给出了相应的结果.

### PointRCNN

请参考 [PointRCNN](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/point_rcnn) 获取更多细节，我们在 KITTI 数据集上给出了相应的结果.

### MonoFlex

请参考 [MonoFlex](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/monoflex) 获取更多细节，我们在 KITTI 数据集上给出了相应的结果.

### Mixed Precision (FP16) Training

细节请参考 [Mixed Precision (FP16) Training] 在 PointPillars 训练的样例 (https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py).
