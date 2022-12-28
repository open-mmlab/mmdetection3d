DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries

> [DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](https://arxiv.org/abs/2110.06922)

<!-- [ALGORITHM] -->

## Abstract

We introduce a framework for multi-camera 3D object detection. In
contrast to existing works, which estimate 3D bounding boxes directly from
monocular images or use depth prediction networks to generate input for 3D object
detection from 2D information, our method manipulates predictions directly
in 3D space. Our architecture extracts 2D features from multiple camera images
and then uses a sparse set of 3D object queries to index into these 2D features,
linking 3D positions to multi-view images using camera transformation matrices.
Finally, our model makes a bounding box prediction per object query, using a
set-to-set loss to measure the discrepancy between the ground-truth and the prediction.
This top-down approach outperforms its bottom-up counterpart in which
object bounding box prediction follows per-pixel depth estimation, since it does
not suffer from the compounding error introduced by a depth prediction model.
Moreover, our method does not require post-processing such as non-maximum
suppression, dramatically improving inference speed. We achieve state-of-the-art
performance on the nuScenes autonomous driving benchmark.

<div align=center>
<img src="https://user-images.githubusercontent.com/67246790/209751755-3d0f0ad5-6a39-4d14-a1c7-346b5c228a1b.png" width="800"/>
</div>

## Introduction

This directory contains the implementations of DETR3D (https://arxiv.org/abs/2110.06922). Our implementations are built on top of MMdetection3D.
We have updated DETR3D to be compatible with latest mmdet3d-dev1.x. The codebase and config files have all changed to adapt to the new mmdet3d version. All previous pretrained models are verified with the result listed below. However, newly trained models are yet to be uploaded.

## Train

1. Downloads the [pretrained backbone weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) to pretrained/

2. For example, to train DETR3D on 8 GPUs, please use

```bash
bash tools/dist_train.sh projects/detr3d/configs/detr3d_res101_gridmask.py 8
```

## Evaluation using pretrained models

1. Download the weights accordingly.

   |                                                   Backbone                                                   | mAP  | NDS  |                                                                                         Download                                                                                         |
   | :----------------------------------------------------------------------------------------------------------: | :--: | :--: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
   |                     [DETR3D, ResNet101 w/ DCN(old)](./configs/detr3d_res101_gridmask.py)                     | 34.7 | 42.2 | [model](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) \| [log](https://drive.google.com/file/d/1uvrf42seV4XbWtir-2XjrdGUZ2Qbykid/view?usp=sharing) |
   |                        [above, + CBGS(old)](./configs/detr3d_res101_gridmask_cbgs.py)                        | 34.9 | 43.4 | [model](https://drive.google.com/file/d/1sXPFiA18K9OMh48wkk9dF1MxvBDUCj2t/view?usp=sharing) \| [log](https://drive.google.com/file/d/1NJNggvFGqA423usKanqbsZVE_CzF4ltT/view?usp=sharing) |
   | [DETR3D, VoVNet on trainval, evaluation on test set(old)](./configs/detr3d_vovnet_gridmask_trainval_cbgs.py) | 41.2 | 47.9 | [model](https://drive.google.com/file/d/1d5FaqoBdUH6dQC3hBKEZLcqbvWK0p9Zv/view?usp=sharing) \| [log](https://drive.google.com/file/d/1ONEMm_2W9MZAutjQk1UzaqRywz5PMk3p/view?usp=sharing) |

2. Convert the old weights
   From v0.17.3 to v1.0.0, mmdet3d has changed its bbox representation. Given that Box(x,y,z,θ), we have x_new = y_old, y_new = x_old, θ_new = -θ_old - π/2.

   Current pretrained models( end with '(old)' ) are in trained on v0.17.3. Our regression branch outputs (cx,cy,w,l,cz,h,sin(θ),cos(θ),vx,vy). For a previous model which outputs y=\[y0,y1,y2,y3,y4,y5,y6,y7,y8,y9\], we get y_new = \[...,y3,y2,...,-y7,-y6\]. So we should change the final Linear layer's weight accordingly.

   To convert the old weights, please use

   `python projects/detr3d/detr3d/old_detr3d_converter.py ckpt/detr3d_resnet101.pth new_ckpt/detr3d_r101_v1.0.0.pth --code_size 10`

3. Testing

   To test, use:

   `bash tools/dist_test.sh projects/detr3d/configs/nuscene/detr3d_res101_gridmask.py ckpts/detr3d_r101_v1.0.0.pth 8 --eval=bbox`

   <!-- Current pretrained models( end with '(old)' ) are in trained on v0.17.3. and we make them compatible with new mmdet3d by rewriting `_load_from_state_dict` method in [`detr3d.py`](./detr3d/detr3d.py) -->

## Citation

If you find this repo useful for your research, please consider citing the papers

```
@inproceedings{
   detr3d,
   title={DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries},
   author={Wang, Yue and Guizilini, Vitor and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and and Solomon, Justin M.},
   booktitle={The Conference on Robot Learning ({CoRL})},
   year={2021}
}
```
