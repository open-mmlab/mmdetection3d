# 3DSSD: Point-based 3D Single Stage Object Detector

> [3DSSD: Point-based 3D Single Stage Object Detector](https://arxiv.org/abs/2002.10187)

<!-- [ALGORITHM] -->

## Abstract

Currently, there have been many kinds of voxel-based 3D single stage detectors, while point-based single stage methods are still underexplored. In this paper, we first present a lightweight and effective point-based 3D single stage object detector, named 3DSSD, achieving a good balance between accuracy and efficiency. In this paradigm, all upsampling layers and refinement stage, which are indispensable in all existing point-based methods, are abandoned to reduce the large computation cost. We novelly propose a fusion sampling strategy in downsampling process to make detection on less representative points feasible. A delicate box prediction network including a candidate generation layer, an anchor-free regression head with a 3D center-ness assignment strategy is designed to meet with our demand of accuracy and speed. Our paradigm is an elegant single stage anchor-free framework, showing great superiority to other existing methods. We evaluate 3DSSD on widely used KITTI dataset and more challenging nuScenes dataset. Our method outperforms all state-of-the-art voxel-based single stage methods by a large margin, and has comparable performance to two stage point-based methods as well, with inference speed more than 25 FPS, 2x faster than former state-of-the-art point-based methods.

<div align=center>
<img src="https://user-images.githubusercontent.com/30491025/143854187-54ed1257-a046-4764-81cd-d2c8404137d3.png" width="800"/>
</div>

## Introduction

We implement 3DSSD and provide the results and checkpoints on KITTI datasets.

Some settings in our implementation are different from the [official implementation](https://github.com/Jia-Research-Lab/3DSSD), which bring marginal differences to the performance on KITTI datasets in our experiments. To simplify and unify the models of our implementation, we skip them in our models. These differences are listed as below:

1. We keep the scenes without any object while the official code skips these scenes in training. In the official implementation, only 3229 and 3394 samples are used as training and validation sets, respectively. In our implementation, we keep using 3712 and 3769 samples as training and validation sets, respectively, as those used for all the other models in our implementation on KITTI datasets.
2. We do not modify the decay of `batch normalization` during training.
3. While using [`DataBaseSampler`](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/dbsampler.py#L80) for data augmentation, the official code uses road planes as reference to place the sampled objects while we do not.
4. We perform detection using LIDAR coordinates while the official code uses camera coordinates.

## Results and models

### KITTI

|                   Backbone                    | Class | Lr schd | Mem (GB) | Inf time (fps) |           mAP            |                                                                                                                                                Download                                                                                                                                                |
| :-------------------------------------------: | :---: | :-----: | :------: | :------------: | :----------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PointNet2SAMSG](./3dssd_4x4_kitti-3d-car.py) |  Car  |   72e   |   4.7    |                | 78.58(81.27)<sup>1</sup> | [model](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/3dssd/3dssd_4x4_kitti-3d-car/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/3dssd/3dssd_4x4_kitti-3d-car/3dssd_4x4_kitti-3d-car_20210818_203828.log.json) |

\[1\]: We report two different 3D object detection performance here. 78.58mAP is evaluated by our evaluation code and 81.27mAP is evaluated by the official development kit （so as that used in the paper and official code of 3DSSD ）. We found that the commonly used Python implementation of [`rotate_iou`](https://github.com/traveller59/second.pytorch/blob/e42e4a0e17262ab7d180ee96a0a36427f2c20a44/second/core/non_max_suppression/nms_gpu.py#L605) which is used in our KITTI dataset evaluation, is different from the official implementation in [KITTI benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

## Citation

```latex
@inproceedings{yang20203dssd,
    author = {Zetong Yang and Yanan Sun and Shu Liu and Jiaya Jia},
    title = {3DSSD: Point-based 3D Single Stage Object Detector},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2020}
}
```
