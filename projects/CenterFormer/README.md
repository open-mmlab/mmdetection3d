# CenterFormer: Center-based Transformer for 3D Object Detection

> [CenterFormer: Center-based Transformer for 3D Object Detection](https://arxiv.org/abs/2209.05588)

<!-- [ALGORITHM] -->

## Abstract

Query-based transformer has shown great potential in con-
structing long-range attention in many image-domain tasks, but has
rarely been considered in LiDAR-based 3D object detection due to the
overwhelming size of the point cloud data. In this paper, we propose
CenterFormer, a center-based transformer network for 3D object de-
tection. CenterFormer first uses a center heatmap to select center candi-
dates on top of a standard voxel-based point cloud encoder. It then uses
the feature of the center candidate as the query embedding in the trans-
former. To further aggregate features from multiple frames, we design
an approach to fuse features through cross-attention. Lastly, regression
heads are added to predict the bounding box on the output center feature
representation. Our design reduces the convergence difficulty and compu-
tational complexity of the transformer structure. The results show signif-
icant improvements over the strong baseline of anchor-free object detec-
tion networks. CenterFormer achieves state-of-the-art performance for a
single model on the Waymo Open Dataset, with 73.7% mAPH on the val-
idation set and 75.6% mAPH on the test set, significantly outperforming
all previously published CNN and transformer-based methods. Our code
is publicly available at https://github.com/TuSimple/centerformer

<div align=center>
<img src="https://user-images.githubusercontent.com/34888372/209500088-b707d7cd-d4d5-4f20-8fdf-a2c7ad15df34.png" width="800"/>
</div>

## Introduction

We implement CenterFormer and provide the results and checkpoints on Waymo dataset.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMDetection3D's root directory, run the following command to train the model:

```bash
python tools/train.py projects/CenterFormer/configs/centerformer_voxel01_second-atten_secfpn-atten_4xb4-cyclic-20e_waymoD5-3d-3class.py
```

For multi-gpu training, run:

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=${NUM_GPUS} --master_port=29506 --master_addr="127.0.0.1" tools/train.py projects/CenterFormer/configs/centerformer_voxel01_second-atten_secfpn-atten_4xb4-cyclic-20e_waymoD5-3d-3class.py
```

### Testing commands

In MMDetection3D's root directory, run the following command to test the model:

```bash
python tools/test.py projects/CenterFormer/configs/centerformer_voxel01_second-atten_secfpn-atten_4xb4-cyclic-20e_waymoD5-3d-3class.py ${CHECKPOINT_PATH}
```

## Results and models

### Waymo

|                                                      Backbone                                                       | Load Interval | Voxel type (voxel size) | Multi-Class NMS | Multi-frames | Mem (GB) | Inf time (fps) | mAP@L1 | mAPH@L1 | mAP@L2 | **mAPH@L2** |                                                                                                                                 Download                                                                                                                                  |
| :-----------------------------------------------------------------------------------------------------------------: | :-----------: | :---------------------: | :-------------: | :----------: | :------: | :------------: | :----: | :-----: | :----: | :---------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [SECFPN_WithAttention](./configs/centerformer_voxel01_second-attn_secfpn-attn_4xb4-cyclic-20e_waymoD5-3d-3class.py) |       5       |       voxel (0.1)       |        ✓        |      ×       |   14.8   |                |  72.2  |  69.5   |  65.9  |    63.3     | [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/centerformer/centerformer_voxel01_second-attn_secfpn-attn_4xb4-cyclic-20e_waymoD5-3d-3class/centerformer_voxel01_second-attn_secfpn-attn_4xb4-cyclic-20e_waymoD5-3d-3class_20221227_205613-70c9ad37.log) |

**Note** that `SECFPN_WithAttention` denotes both SECOND and SECONDFPN with ChannelAttention and SpatialAttention.

## Citation

```latex
@InProceedings{Zhou_centerformer,
title = {CenterFormer: Center-based Transformer for 3D Object Detection},
author = {Zhou, Zixiang and Zhao, Xiangchen and Wang, Yu and Wang, Panqu and Foroosh, Hassan},
booktitle = {ECCV},
year = {2022}
}
```
