# DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets

> [DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets](https://arxiv.org/abs/2301.06051)

<!-- [ALGORITHM] -->

## Abstract

Designing an efficient yet deployment-friendly 3D backbone to handle sparse point clouds is a fundamental problem
in 3D perception. Compared with the customized sparse
convolution, the attention mechanism in Transformers is
more appropriate for flexibly modeling long-range relationships and is easier to be deployed in real-world applications.
However, due to the sparse characteristics of point clouds,
it is non-trivial to apply a standard transformer on sparse
points. In this paper, we present Dynamic Sparse Voxel
Transformer (DSVT), a single-stride window-based voxel
Transformer backbone for outdoor 3D perception. In order
to efficiently process sparse points in parallel, we propose
Dynamic Sparse Window Attention, which partitions a series
of local regions in each window according to its sparsity
and then computes the features of all regions in a fully parallel manner. To allow the cross-set connection, we design
a rotated set partitioning strategy that alternates between
two partitioning configurations in consecutive self-attention
layers. To support effective downsampling and better encode geometric information, we also propose an attentionstyle 3D pooling module on sparse points, which is powerful
and deployment-friendly without utilizing any customized
CUDA operations. Our model achieves state-of-the-art performance with a broad range of 3D perception tasks. More
importantly, DSVT can be easily deployed by TensorRT with
real-time inference speed (27Hz). Code will be available at
https://github.com/Haiyang-W/DSVT.

<div align=center>
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/34888372/245692705-e61be20c-2a7d-4ab9-85e3-b36f662c1bdf.png" width="800"/>
</div>

## Introduction

We implement DSVT and provide the results on Waymo dataset.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Installation

```shell
pip install torch_scatter==2.0.9
python projects/DSVT/setup.py develop # compile `ingroup_inds_op` cuda operation
```

### Testing commands

In MMDetection3D's root directory, run the following command to test the model:

```bash
python tools/test.py projects/DSVT/configs/dsvt_voxel032_res-second_secfpn_8xb1-cyclic-12e_waymoD5-3d-3class.py ${CHECKPOINT_PATH}
```

### Training commands

In MMDetection3D's root directory, run the following command to test the model:

```bash
tools/dist_train.sh projects/DSVT/configs/dsvt_voxel032_res-second_secfpn_8xb1-cyclic-12e_waymoD5-3d-3class.py 8 --sync_bn torch
```

## Results and models

### Waymo

|                                     Middle Encoder                                     |                                          Backbone                                           | Load Interval | Voxel type (voxel size) | Multi-Class NMS | Multi-frames | mAP@L1 | mAPH@L1 | mAP@L2 | **mAPH@L2** |                                                                           Download                                                                           |
| :------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: | :-----------: | :---------------------: | :-------------: | :----------: | :----: | :-----: | :----: | :---------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [DSVT](./configs/dsvt_voxel032_res-second_secfpn_8xb1-cyclic-12e_waymoD5-3d-3class.py) | [ResSECOND](./configs/dsvt_voxel032_res-second_secfpn_8xb1-cyclic-12e_waymoD5-3d-3class.py) |       5       |      voxel (0.32)       |        ✓        |      ×       |  75.5  |  72.4   |  69.2  |    66.3     | [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/dsvt/dsvt_voxel032_res-second_secfpn_8xb1-cyclic-12e_waymoD5-3d-3class_20230917_102130.log) |

**Note**:

- `ResSECOND` denotes the base block in SECOND has residual layers.

- Regrettably, we are unable to provide the pre-trained model weights due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/), so we only provide the training logs as shown above.

## Citation

```latex
@inproceedings{wang2023dsvt,
    title={DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets},
    author={Haiyang Wang, Chen Shi, Shaoshuai Shi, Meng Lei, Sen Wang, Di He, Bernt Schiele and Liwei Wang},
    booktitle={CVPR},
    year={2023}
}
```
