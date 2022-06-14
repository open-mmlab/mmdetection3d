# H3DNet: 3D Object Detection Using Hybrid Geometric Primitives

> [H3DNet: 3D Object Detection Using Hybrid Geometric Primitives](https://arxiv.org/abs/2006.05682)

<!-- [ALGORITHM] -->

## Abstract

We introduce H3DNet, which takes a colorless 3D point cloud as input and outputs a collection of oriented object bounding boxes (or BB) and their semantic labels. The critical idea of H3DNet is to predict a hybrid set of geometric primitives, i.e., BB centers, BB face centers, and BB edge centers. We show how to convert the predicted geometric primitives into object proposals by defining a distance function between an object and the geometric primitives. This distance function enables continuous optimization of object proposals, and its local minimums provide high-fidelity object proposals. H3DNet then utilizes a matching and refinement module to classify object proposals into detected objects and fine-tune the geometric parameters of the detected objects. The hybrid set of geometric primitives not only provides more accurate signals for object detection than using a single type of geometric primitives, but it also provides an overcomplete set of constraints on the resulting 3D layout. Therefore, H3DNet can tolerate outliers in predicted geometric primitives. Our model achieves state-of-the-art 3D detection results on two large datasets with real 3D scans, ScanNet and SUN RGB-D.

<div align=center>
<img src="https://user-images.githubusercontent.com/36950400/143868884-26f7fc63-93fd-48cb-a469-e2f55fda5550.png" width="800"/>
</div>

## Introduction

We implement H3DNet and provide the result and checkpoints on ScanNet datasets.

## Results and models

### ScanNet

|                      Backbone                       | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 | AP@0.5 |                                                                                                                                                           Download                                                                                                                                                           |
| :-------------------------------------------------: | :-----: | :------: | :------------: | :-----: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MultiBackbone](./h3dnet_3x8_scannet-3d-18class.py) |   3x    |   7.9    |                |  66.07  | 47.68  | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/h3dnet/h3dnet_scannet-3d-18class/h3dnet_3x8_scannet-3d-18class_20210824_003149-414bd304.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/h3dnet/h3dnet_scannet-3d-18class/h3dnet_3x8_scannet-3d-18class_20210824_003149.log.json) |

**Notice**: If your current mmdetection3d version >= 0.6.0, and you are using the checkpoints downloaded from the above links or using checkpoints trained with mmdetection3d version \< 0.6.0, the checkpoints have to be first converted via [tools/model_converters/convert_h3dnet_checkpoints.py](../../tools/model_converters/convert_h3dnet_checkpoints.py):

```
python ./tools/model_converters/convert_h3dnet_checkpoints.py ${ORIGINAL_CHECKPOINT_PATH} --out=${NEW_CHECKPOINT_PATH}
```

Then you can use the converted checkpoints following [getting_started.md](../../docs/en/getting_started.md).

## Citation

```latex
@inproceedings{zhang2020h3dnet,
    author = {Zhang, Zaiwei and Sun, Bo and Yang, Haitao and Huang, Qixing},
    title = {H3DNet: 3D Object Detection Using Hybrid Geometric Primitives},
    booktitle = {Proceedings of the European Conference on Computer Vision},
    year = {2020}
}
```
