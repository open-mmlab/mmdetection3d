# Deep Hough Voting for 3D Object Detection in Point Clouds

## Introduction

<!-- [ALGORITHM] -->

We implement VoteNet and provide the result and checkpoints on ScanNet and SUNRGBD datasets.

```
@inproceedings{qi2019deep,
    author = {Qi, Charles R and Litany, Or and He, Kaiming and Guibas, Leonidas J},
    title = {Deep Hough Voting for 3D Object Detection in Point Clouds},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
    year = {2019}
}
```

## Results

### ScanNet

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 |AP@0.5| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [PointNet++](./votenet_8x8_scannet-3d-18class.py)     |  3x    |4.1||62.90|39.91|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/votenet/votenet_8x8_scannet-3d-18class/votenet_8x8_scannet-3d-18class_20200620_230238-2cea9c3a.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/votenet/votenet_8x8_scannet-3d-18class/votenet_8x8_scannet-3d-18class_20200620_230238.log.json)|

### SUNRGBD

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 |AP@0.5| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [PointNet++](./votenet_16x8_sunrgbd-3d-10class.py)     |  3x    |8.1||59.07|35.77|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/votenet/votenet_16x8_sunrgbd-3d-10class/votenet_16x8_sunrgbd-3d-10class_20200620_230238-4483c0c0.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/votenet/votenet_16x8_sunrgbd-3d-10class/votenet_16x8_sunrgbd-3d-10class_20200620_230238.log.json)|

**Notice**: If your current mmdetection3d version >= 0.6.0, and you are using the checkpoints downloaded from the above links or using checkpoints trained with mmdetection3d version < 0.6.0, the checkpoints have to be first converted via [tools/model_converters/convert_votenet_checkpoints.py](../../tools/model_converters/convert_votenet_checkpoints.py):

```
python ./tools/model_converters/convert_votenet_checkpoints.py ${ORIGINAL_CHECKPOINT_PATH} --out=${NEW_CHECKPOINT_PATH}
```

Then you can use the converted checkpoints following [getting_started.md](../../docs/getting_started.md).

## Indeterminism

Since test data preparation randomly downsamples the points, and the test script uses fixed random seeds while the random seeds of validation in training are not fixed, the test results may be slightly different from the results reported above.

## IoU loss

Adding IoU loss (simply = 1-IoU) boosts VoteNet's performance. To use IoU loss, add this loss term to the config file:

```python
iou_loss=dict(type='AxisAlignedIoULoss', reduction='sum', loss_weight=10.0 / 3.0)
```

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 |AP@0.5| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [PointNet++](./votenet_iouloss_8x8_scannet-3d-18class.py)     |  3x    |4.1||63.81|44.21|/|

For now, we only support calculating IoU loss for axis-aligned bounding boxes since the CUDA op of general 3D IoU calculation does not implement the backward method. Therefore, IoU loss can only be used for ScanNet dataset for now.
