# Model Zoo

## Common settings

- We use distributed training.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.
- We report the inference time as the total time of network forwarding and post-processing, excluding the data loading time. Results are obtained with the script [benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/benchmark.py) which computes the average time on 2000 images.

## Baselines

### SECOND

Please refer to [SECOND](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/second) for details. We provide SECOND baselines on KITTI and Waymo datasets.

### PointPillars

Please refer to [PointPillars](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars) for details. We provide pointpillars baselines on KITTI, nuScenes, Lyft, and Waymo datasets.

### Part-A2

Please refer to [Part-A2](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/parta2) for details.

### VoteNet

Please refer to [VoteNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/votenet) for details. We provide VoteNet baselines on ScanNet and SUNRGBD datasets.

### Dynamic Voxelization

Please refer to [Dynamic Voxelization](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/dynamic_voxelization) for details.

### MVXNet

Please refer to [MVXNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/mvxnet) for details.

### RegNetX

Please refer to [RegNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/regnet) for details. We provide pointpillars baselines with RegNetX backbones on nuScenes and Lyft datasets currently.

### nuImages

We also support baseline models on [nuImages dataset](https://www.nuscenes.org/nuimages). Please refer to [nuImages](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/nuimages) for details. We report Mask R-CNN, Cascade Mask R-CNN and HTC results currently.

### H3DNet

Please refer to [H3DNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/h3dnet) for details.

### 3DSSD

Please refer to [3DSSD](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/3dssd) for details.

### CenterPoint

Please refer to [CenterPoint](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/centerpoint) for details.

### SSN

Please refer to [SSN](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/ssn) for details. We provide pointpillars with shape-aware grouping heads used in SSN on the nuScenes and Lyft datasets currently.

### ImVoteNet

Please refer to [ImVoteNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/imvotenet) for details. We provide ImVoteNet baselines on SUNRGBD dataset.

### FCOS3D

Please refer to [FCOS3D](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/fcos3d) for details. We provide FCOS3D baselines on the nuScenes dataset.

### PointNet++

Please refer to [PointNet++](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointnet2) for details. We provide PointNet++ baselines on ScanNet and S3DIS datasets.

### Group-Free-3D

Please refer to [Group-Free-3D](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/groupfree3d) for details. We provide Group-Free-3D baselines on ScanNet dataset.

### ImVoxelNet

Please refer to [ImVoxelNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/imvoxelnet) for details. We provide ImVoxelNet baselines on KITTI dataset.

### PAConv

Please refer to [PAConv](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/paconv) for details. We provide PAConv baselines on S3DIS dataset.

### DGCNN

Please refer to [DGCNN](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/dgcnn) for details. We provide DGCNN baselines on S3DIS dataset.

### SMOKE

Please refer to [SMOKE](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/smoke) for details. We provide SMOKE baselines on KITTI dataset.

### PGD

Please refer to [PGD](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/pgd) for details. We provide PGD baselines on KITTI and nuScenes dataset.

### PointRCNN

Please refer to [PointRCNN](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/point_rcnn) for details. We provide PointRCNN baselines on KITTI dataset.

### MonoFlex

Please refer to [MonoFlex](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/monoflex) for details. We provide MonoFlex baselines on KITTI dataset.

### SA-SSD

Please refer to [SA-SSD](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/sassd) for details. We provide SA-SSD baselines on the KITTI dataset.

### FCAF3D

Please refer to [FCAF3D](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/fcaf3d) for details. We provide FCAF3D baselines on the ScanNet, S3DIS, and SUN RGB-D datasets.

### PV-RCNN

Please refer to [PV-RCNN](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/pv_rcnn) for details. We provide PV-RCNN baselines on the KITTI dataset.

### BEVFusion

Please refer to [BEVFusion](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/BEVFusion) for details. We provide BEVFusion baselines on the NuScenes dataset.

### CenterFormer

Please refer to [CenterFormer](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/CenterFormer) for details. We provide CenterFormer baselines on the Waymo dataset.

### TR3D

Please refer to [TR3D](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/TR3D) for details. We provide TR3D baselines on the ScanNet, SUN RGB-D and S3DIS dataset.

### DETR3D

Please refer to [DETR3D](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/DETR3D) for details. We provide DETR3D baselines on the nuScenes dataset.

### PETR

Please refer to [PETR](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/PETR) for details. We provide PETR baselines on the nuScenes dataset.

### TPVFormer

Please refer to [TPVFormer](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/TPVFormer) for details. We provide TPVFormer baselines on the nuScenes dataset.

### Mixed Precision (FP16) Training

Please refer to [Mixed Precision (FP16) Training on PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py) for details.
