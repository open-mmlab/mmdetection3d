# DETR3D

This directory contains the implementations of DETR3D (https://arxiv.org/abs/2110.06922). Our implementations are built on top of MMdetection3D.  

### Prerequisite

1. mmcv (https://github.com/open-mmlab/mmcv)

2. mmdet (https://github.com/open-mmlab/mmdetection)

3. mmseg (https://github.com/open-mmlab/mmsegmentation)

4. mmdet3d-v1.0.0 (https://github.com/open-mmlab/mmdetection3d)

### Data
1. Follow the mmdet3d to process the data.

### Train
1. Downloads the [pretrained backbone weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) to pretrained/ 

2. For example, to train DETR3D on 8 GPUs, please use

`bash tools/dist_train.sh projects/detr3d/configs/nuscene/detr3d_res101_gridmask.py 8`

### Evaluation using pretrained models
1. Download the weights accordingly.

|  Backbone   | mAP | NDS | Download |
| :---------: | :----: |:----: | :------: |
|[DETR3D, ResNet101 w/ DCN(old)](./projects/configs/detr3d/detr3d_res101_gridmask.py)|34.7|42.2|[model](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1uvrf42seV4XbWtir-2XjrdGUZ2Qbykid/view?usp=sharing)|
|[above, + CBGS(old)](./projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py)|34.9|43.4|[model](https://drive.google.com/file/d/1sXPFiA18K9OMh48wkk9dF1MxvBDUCj2t/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1NJNggvFGqA423usKanqbsZVE_CzF4ltT/view?usp=sharing)|
|[DETR3D, VoVNet on trainval, evaluation on test set(old)](./projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py)| 41.2 | 47.9 |[model](https://drive.google.com/file/d/1d5FaqoBdUH6dQC3hBKEZLcqbvWK0p9Zv/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1ONEMm_2W9MZAutjQk1UzaqRywz5PMk3p/view?usp=sharing)|


2. Testing
From v0.17.3 to v1.0.0, mmdet3d has changed its bbox representation. Given that Box(x,y,z,θ), we have x_new = y_old, y_new = x_old, θ_new = -θ_old - π/2.

   Currently pretrained models( end with '(old)' ) are in trained on v0.17.3.

   To compatible with older models, you may test using

   `bash tools/dist_test.sh projects/detr3d/configs/nuscene/detr3d_r101_test_old_model.py ckpts/detr3d_resnet101.pth 8 --eval=bbox`

   To test new models:

   `bash tools/dist_test.sh projects/detr3d/configs/nuscene/detr3d_res101_gridmask.py ckpts/detr3d_r101_v1.0.0-rc2.pth 8 --eval=bbox`


 
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
