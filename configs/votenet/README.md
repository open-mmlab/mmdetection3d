# Deep Hough Voting for 3D Object Detection in Point Clouds

> [Deep Hough Voting for 3D Object Detection in Point Clouds](https://arxiv.org/abs/1904.09664)

<!-- [ALGORITHM] -->

## Abstract

Current 3D object detection methods are heavily influenced by 2D detectors. In order to leverage architectures in 2D detectors, they often convert 3D point clouds to regular grids (i.e., to voxel grids or to bird's eye view images), or rely on detection in 2D images to propose 3D boxes. Few works have attempted to directly detect objects in point clouds. In this work, we return to first principles to construct a 3D detection pipeline for point cloud data and as generic as possible. However, due to the sparse nature of the data -- samples from 2D manifolds in 3D space -- we face a major challenge when directly predicting bounding box parameters from scene points: a 3D object centroid can be far from any surface point thus hard to regress accurately in one step. To address the challenge, we propose VoteNet, an end-to-end 3D object detection network based on a synergy of deep point set networks and Hough voting. Our model achieves state-of-the-art 3D detection on two large datasets of real 3D scans, ScanNet and SUN RGB-D with a simple design, compact model size and high efficiency. Remarkably, VoteNet outperforms previous methods by using purely geometric information without relying on color images.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/143888295-af7435b4-9f75-4669-b5f8-a19ae24a051c.png" width="800"/>
</div>

## Introduction

We implement VoteNet and provide the result and checkpoints on ScanNet and SUNRGBD datasets.

## Results and models

### ScanNet

|                     Backbone                      | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 | AP@0.5 |                                                                                                                                                                  Download                                                                                                                                                                  |
| :-----------------------------------------------: | :-----: | :------: | :------------: | :-----: | :----: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PointNet++](./votenet_8x8_scannet-3d-18class.py) |   3x    |   4.1    |                |  62.34  | 40.82  | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/votenet/votenet_8x8_scannet-3d-18class/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/votenet/votenet_8x8_scannet-3d-18class/votenet_8x8_scannet-3d-18class_20210823_234503.log.json) |

### SUNRGBD

|                      Backbone                      | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 | AP@0.5 |                                                                                                                                                                    Download                                                                                                                                                                    |
| :------------------------------------------------: | :-----: | :------: | :------------: | :-----: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PointNet++](./votenet_16x8_sunrgbd-3d-10class.py) |   3x    |   8.1    |                |  59.78  | 35.77  | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/votenet/votenet_16x8_sunrgbd-3d-10class/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/votenet/votenet_16x8_sunrgbd-3d-10class/votenet_16x8_sunrgbd-3d-10class_20210820_162823.log.json) |

**Notice**: If your current mmdetection3d version >= 0.6.0, and you are using the checkpoints downloaded from the above links or using checkpoints trained with mmdetection3d version \< 0.6.0, the checkpoints have to be first converted via [tools/model_converters/convert_votenet_checkpoints.py](../../tools/model_converters/convert_votenet_checkpoints.py):

```
python ./tools/model_converters/convert_votenet_checkpoints.py ${ORIGINAL_CHECKPOINT_PATH} --out=${NEW_CHECKPOINT_PATH}
```

Then you can use the converted checkpoints following [getting_started.md](../../docs/en/getting_started.md).

## Indeterminism

Since test data preparation randomly downsamples the points, and the test script uses fixed random seeds while the random seeds of validation in training are not fixed, the test results may be slightly different from the results reported above.

## IoU loss

Adding IoU loss (simply = 1-IoU) boosts VoteNet's performance. To use IoU loss, add this loss term to the config file:

```python
iou_loss=dict(type='AxisAlignedIoULoss', reduction='sum', loss_weight=10.0 / 3.0)
```

|                         Backbone                          | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 | AP@0.5 | Download |
| :-------------------------------------------------------: | :-----: | :------: | :------------: | :-----: | :----: | :------: |
| [PointNet++](./votenet_iouloss_8x8_scannet-3d-18class.py) |   3x    |   4.1    |                |  63.81  | 44.21  |    /     |

For now, we only support calculating IoU loss for axis-aligned bounding boxes since the CUDA op of general 3D IoU calculation does not implement the backward method. Therefore, IoU loss can only be used for ScanNet dataset for now.

## Citation

```latex
@inproceedings{qi2019deep,
    author = {Qi, Charles R and Litany, Or and He, Kaiming and Guibas, Leonidas J},
    title = {Deep Hough Voting for 3D Object Detection in Point Clouds},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
    year = {2019}
}
```
