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

## Results

### KITTI
|  Backbone   |Class| Lr schd | Mem (GB) | Inf time (fps) | mAP |Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [SECFPN](./3dssd_kitti-3d-car.py)| Car |AdamW 72e|4.7||78.35||
