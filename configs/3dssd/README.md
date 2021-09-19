# 3DSSD: Point-based 3D Single Stage Object Detector

## Introduction

<!-- [ALGORITHM] -->

We implement 3DSSD and provide the results and checkpoints on KITTI datasets.

```
@inproceedings{yang20203dssd,
    author = {Zetong Yang and Yanan Sun and Shu Liu and Jiaya Jia},
    title = {3DSSD: Point-based 3D Single Stage Object Detector},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2020}
}
```

### Experiment details on KITTI datasets

Some settings in our implementation are different from the [official implementation](https://github.com/Jia-Research-Lab/3DSSD), which bring marginal differences to the performance on KITTI datasets in our experiments. To simplify and unify the models of our implementation, we skip them in our models. These differences are listed as below:
1. We keep the scenes without any object while the official code skips these scenes in training. In the official implementation, only 3229 and 3394 samples are used as training and validation sets, respectively. In our implementation, we keep using 3712 and 3769 samples as training and validation sets, respectively, as those used for all the other models in our implementation on KITTI datasets.
2. We do not modify the decay of `batch normalization` during training.
3. While using [`DataBaseSampler`](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/dbsampler.py#L80) for data augmentation, the official code uses road planes as reference to place the sampled objects while we do not.
4. We perform detection using LIDAR coordinates while the official code uses camera coordinates.

## Results

### KITTI

|  Backbone   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP |Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [PointNet2SAMSG](./3dssd_4x4_kitti-3d-car.py)| Car |72e|4.7||78.69(81.27)<sup>1</sup>|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/3dssd/3dssd_kitti-3d-car_20210602_124438-b4276f56.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/3dssd/3dssd_kitti-3d-car_20210602_124438.log.json)|

[1]: We report two different 3D object detection performance here. 78.69mAP is evaluated by our evaluation code and 81.27mAP is evaluated by the official development kit （so as that used in the paper and official code of 3DSSD ）. We found that the commonly used Python implementation of [`rotate_iou`](https://github.com/traveller59/second.pytorch/blob/e42e4a0e17262ab7d180ee96a0a36427f2c20a44/second/core/non_max_suppression/nms_gpu.py#L605) which is used in our KITTI dataset evaluation, is different from the official implemention in [KITTI benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
