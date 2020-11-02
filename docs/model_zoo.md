# Model Zoo

## Common settings

- We use distributed training.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.
- We report the inference time as the total time of network forwarding and post-processing, excluding the data loading time. Results are obtained with the script [benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/benchmark.py) which computes the average time on 2000 images.

## Baselines

### SECOND

Please refer to [SECOND](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/second) for details. We provide SECOND baselines on KITTI and Waymo datasets.

### PointPillars

Please refer to [PointPillars](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars) for details. We provide pointpillars baselines on KITTI, nuScenes, Lyft, and Waymo datasets.

### Part-A2

Please refer to [Part-A2](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/parta2) for details.

### VoteNet

Please refer to [VoteNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/votenet) for details. We provide VoteNet baselines on ScanNet and SUNRGBD datasets.

### Dynamic Voxelization

Please refer to [Dynamic Voxelization](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/dynamic_voxelization) for details.

### MVXNet

Please refer to [MVXNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/mvxnet) for details.

### RegNetX
Please refer to [RegNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/regnet) for details. We provide pointpillars baselines with RegNetX backbones on nuScenes and Lyft datasets currently.

### nuImages
We also support baseline models on [nuImages dataset](https://www.nuscenes.org/nuimages). Please refer to [nuImages](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/nuimages) for details. We report Mask R-CNN, Cascade Mask R-CNN and HTC results currently.

### H3DNet

Please refer to [H3DNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/h3dnet) for details.

### 3DSSD

Please refer to [3DSSD](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/3dssd) for details.

### CenterPoint

Please refer to [CenterPoint](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/centerpoint) for details.

### SSN

Please refer to [SSN](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/ssn) for details. We provide pointpillars with shape-aware grouping heads used in SSN on the nuScenes and Lyft dataset currently.
