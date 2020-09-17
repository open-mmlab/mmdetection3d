# 3DSSD: Point-based 3D Single Stage Object Detector

## Introduction
We implement 3DSSD and provide the result and checkpoints on KITTI datasets.

```
@inproceedings{yang20203dssd,
    author = {Zetong Yang and Yanan Sun and Shu Liu and Jiaya Jia},
    title = {3DSSD: Point-based 3D Single Stage Object Detector},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2020}
}
```

### Experiment details on KITTI datasets
Some experiment settings are different from our implementation and the 3DSSD official code and we found that these differences do no harm to the performance on KITTI datasets. The differences are listed as follows:
1. We keep the scenes that without any object while the official code remove those scenes while preprocessing the dataset.
2. We do not use the 'batch normalization ' decay in training procedure.
3. While using [`DataBaseSampler` ](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/dbsampler.py#L80) for data augmentation, the official code uses road planes as reference to place the sampled objects while we ignore the this extra information.
4. We use LIDAR coordinates while the official code uses Camera coordinates for input points.

## Results

### KITTI
|  Backbone   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP |Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [PointNet2SAMSG](./3dssd_kitti-3d-car.py)| Car |72e|4.7||78.35(80.42)<sup>1</sup>||

[1]: We report two different 3D object detection performance here. 78.35mAP is evaluated by our evaluation code and 80.42mAP is evaluated by the official development kit （so as that used in the paper and official code of 3DSSD ）. We found that the commonly used Python implementation of [`rotate_iou`](https://github.com/traveller59/second.pytorch/blob/e42e4a0e17262ab7d180ee96a0a36427f2c20a44/second/core/non_max_suppression/nms_gpu.py#L605) which is used in our KITTI dataset evaluation, is different from the official implemention in [KITTI BENCHMARK](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
