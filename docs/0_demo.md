# Demo

## Introduction

We provide scipts for multi-modality/single-modality and indoor/outdoor 3D detection demos. The pre-trained models can be downloaded from [model zoo](../docs/model_zoo.md). We provide pre-processed sample data from KITTI and SUN RGB-D dataset. You can use any other data following our pre-processing steps.

## Testing

### Single-modality demo

To test a 3D detector on point cloud data, simply run:

```shell
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}]
```

The visualization results including a point cloud and predicted 3D bounding boxes will be saved in ```demo/PCD_NAME```, which you can open using [MeshLab](http://www.meshlab.net/).

Example on KITTI data using [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) model:

```shell
python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth
```

Example on SUN RGB-D data using [VoteNet](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/votenet) model:

```shell
python demo/pcd_demo.py demo/data/sunrgbd/sunrgbd_000017.bin configs/votenet/votenet_16x8_sunrgbd-3d-10class.py checkpoints/votenet_16x8_sunrgbd-3d-10class_20200620_230238-4483c0c0.pth
```

Remember to convert the VoteNet checkpoint if you are using mmdetection3d version >= 0.6.0. See its [README](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/votenet/README.md) for detailed instructions on how to convert the checkpoint.

### Multi-modality demo

To test a 3D detector on multi-modality data (typically point cloud and image), simply run:

```shell
python demo/multi_modality_demo.py ${PCD_FILE} ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}]
```

where the ```ANNOTATION_FILE``` should provide the 3D to 2D projection matrix. The visualization results including a point cloud, an image, predicted 3D bounding boxes and their projection on the image will be saved in ```demo/PCD_NAME```.

Example on KITTI data using [MVX-Net](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/mvxnet) model:

```shell
python demo/multi_modality_demo.py demo/data/kitti/kitti_000008.bin demo/data/kitti/kitti_000008.png demo/data/kitti/kitti_000008_infos.pkl configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth
```

Example on SUN RGB-D data using [ImVoteNet](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/imvotenet) model:

```shell
python demo/multi_modality_demo.py demo/data/sunrgbd/sunrgbd_000017.bin demo/data/sunrgbd/sunrgbd_000017.jpg demo/data/sunrgbd/sunrgbd_000017_infos.pkl configs/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class.py checkpoints/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210323_184021-d44dcb66.pth
```
