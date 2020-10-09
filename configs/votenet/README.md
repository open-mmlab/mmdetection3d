# Deep Hough Voting for 3D Object Detection in Point Clouds

## Introduction
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
