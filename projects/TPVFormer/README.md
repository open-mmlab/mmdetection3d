# Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction

> [Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2302.07817)

<!-- [ALGORITHM] -->

## Abstract

Modern methods for vision-centric autonomous driving perception widely adopt the bird's-eye-view (BEV) representation to describe a 3D scene. Despite its better efficiency than voxel representation, it has difficulty describing the fine-grained 3D structure of a scene with a single plane. To address this, we propose a tri-perspective view (TPV) representation which accompanies BEV with two additional perpendicular planes. We model each point in the 3D space by summing its projected features on the three planes. To lift image features to the 3D TPV space, we further propose a transformer-based TPV encoder (TPVFormer) to obtain the TPV features effectively. We employ the attention mechanism to aggregate the image features corresponding to each query in each TPV plane. Experiments show that our model trained with sparse supervision effectively predicts the semantic occupancy for all voxels. We demonstrate for the first time that using only camera inputs can achieve comparable performance with LiDAR-based methods on the LiDAR segmentation task on nuScenes. Code: https://github.com/wzzheng/TPVFormer.

<div align=center>
<img src="https://github.com/traveller59/spconv/assets/72679458/8cc8caa6-b330-4f32-9599-3811dc5d7332" width="800"/>
</div>

## Introduction

We implement TPVFormer and provide the results and checkpoints on nuScenes dataset.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMDetection3D's root directory, run the following command to train the model:

1. Downloads the [pretrained backbone weights](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/tpvformer/tpvformer_8xb1-2x_nus-seg/tpvformer_pretrained_fcos3d_r101_dcn.pth) to checkpoints/

2. For example, to train TPVFormer on 8 GPUs, please use

```bash
bash tools/dist_train.sh projects/TPVFormer/config/tpvformer_8xb1-2x_nus-seg.py 8
```

### Testing commands

In MMDetection3D's root directory, run the following command to test the model on 8 GPUs:

```bash
bash tools/dist_test.sh projects/TPVFormer/config/tpvformer_8xb1-2x_nus-seg.py  ${CHECKPOINT_PATH} 8
```

## Results and models

### nuScenes

| Backbone                                                                                                                                         | Neck | Mem (GB) | Inf time (fps) | mIoU | Downloads                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ---- | -------- | -------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ResNet101 w/ DCN](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py) | FPN  | 32.0     | -              | 68.9 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/tpvformer/tpvformer_8xb1-2x_nus-seg/tpvformer_8xb1-2x_nus-seg_20230411_150639-bd3844e2.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/tpvformer/tpvformer_8xb1-2x_nus-seg/tpvformer_8xb1-2x_nus-seg_20230411_150639.log) |

## Citation

```latex
@article{huang2023tri,
    title={Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction},
    author={Huang, Yuanhui and Zheng, Wenzhao and Zhang, Yunpeng and Zhou, Jie and Lu, Jiwen },
    journal={arXiv preprint arXiv:2302.07817},
    year={2023}
}
```
