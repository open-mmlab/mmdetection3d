# NeRF-Det: Learning Geometry-Aware Volumetric Representation for Multi-View 3D Object Detection

> [NeRF-Det: Learning Geometry-Aware Volumetric Representation for Multi-View 3D Object Detection](https://arxiv.org/abs/2307.14620)

<!-- [ALGORITHM] -->

## Abstract

NeRF-Det is a novel method for indoor 3D detection with posed RGB images as input. Unlike existing indoor 3D detection methods that struggle to model scene geometry, NeRF-Det makes novel use of NeRF in an end-to-end manner to explicitly estimate 3D geometry, thereby improving 3D detection performance. Specifically, to avoid the significant extra latency associated with per-scene optimization of NeRF, NeRF-Det introduce sufficient geometry priors to enhance the generalizability of NeRF-MLP. Furthermore, it subtly connect the detection and NeRF branches through a shared MLP, enabling an efficient adaptation of NeRF to detection and yielding geometry-aware volumetric representations for 3D detection. NeRF-Det outperforms state-of-the-arts by 3.9 mAP and 3.1 mAP on the ScanNet and ARKITScenes benchmarks, respectively. The author provide extensive analysis to shed light on how NeRF-Det works. As a result of joint-training design,  NeRF-Det is able to generalize well to unseen scenes for object detection, view synthesis, and depth estimation tasks without requiring per-scene optimization. Code will be available at https://github.com/facebookresearch/NeRF-Det

<div align=center>
<img src="https://chenfengxu714.github.io/nerfdet/static/images/method-cropped_1.png" width="800"/>
</div>

## Introduction

This directory contains the implementations of NeRF-Det (https://arxiv.org/abs/2307.14620). Our implementations are built on top of MMdetection3D. We have updated NeRF-Det to be compatible with latest mmdet3d version. The codebase and config files have all changed to adapt to the new mmdet3d version. All previous pretrained models are verified with the result listed below. However, newly trained models are yet to be uploaded.

<!-- Share any information you would like others to know. For example:
Author: @xxx.
This is an implementation of \[XXX\]. -->

## Dataset

The format of the scannet dataset in the latest version of mmdet3d only supports the lidar tasks. For NeRF-Det, we need to create the new format of ScanNet Dataset.

Please following the files in mmdet3d to prepare the raw data of ScanNet. After that, please use this command to generate the pkls used in nerfdet.

```bash
python projects/NeRF-Det/prepare_infos.py --root-path ./data/scannet --out-dir ./data/scannet
```

The new format of the pkl is organized as below:

- scannet_infos_train.pkl: The train data infos, the detailed info of each scan is as follows:
  - info\['instances'\]:A list of dict contains all annotations, each dict contains all annotation information of single instance.For the i-th instance:
    - info\['instances'\]\[i\]\['bbox_3d'\]: List of 6 numbers representing the axis_aligned in depth coordinate system, in (x,y,z,l,w,h) order.
    - info\['instances'\]\[i\]\['bbox_label_3d'\]: The label of each 3d bounding boxes.
  - info\['cam2img'\]: The intrinsic matrix.Every scene has one matrix.
  - info\['lidar2cam'\]: The extrinsic matrixes.Every scene has 300 matrixes.
  - info\['img_paths'\]: The paths of the 300 rgb pictures.
  - info\['axis_align_matrix'\]: The align matrix.Every scene has one matrix.

After preparing your scannet dataset pkls,please change the paths in configs to fit your project.

## Train

In MMDet3D's root directory, run the following command to train the model:

```bash
python tools/train.py projects/NeRF-Det/configs/nerfdet_res50_2x_low_res.py ${WORK_DIR}
```

## Results and Models

### NeRF-Det

|                            Backbone                             | mAP@25 | mAP@50 |    Log    |
| :-------------------------------------------------------------: | :----: | :----: | :-------: |
|      [NeRF-Det-R50](./configs/nerfdet_res50_2x_low_res.py)      |  53.0  |  26.8  | [log](<>) |
|  [NeRF-Det-R50\*](./configs/nerfdet_res50_2x_low_res_depth.py)  |  52.2  |  28.5  | [log](<>) |
| [NeRF-Det-R101\*](./configs/nerfdet_res101_2x_low_res_depth.py) |  52.3  |  28.5  | [log](<>) |

(Here NeRF-Det-R50\* means this model uses depth information in the training step)

### Notes

- The values showed in the chart all represents the best mAP in the training.

- Since there is a lot of randomness in the behavior of the model, we conducted three experiments on each config and took the average. The mAP showed on the above chart are all average values.

- We also conducted the same experiments in the original code, the results are showed below.

  |    Backbone     | mAP@25 | mAP@50 |
  | :-------------: | :----: | :----: |
  |  NeRF-Det-R50   |  52.8  |  26.8  |
  | NeRF-Det-R50\*  |  52.4  |  27.5  |
  | NeRF-Det-R101\* |  52.8  |  28.6  |

- Attention: Because of the randomness in the construction of the ScanNet dataset itself and the behavior of the model, the training results will fluctuate considerably. According to experimental results and experience, the experimental results will fluctuate by plus or minus 1.5 points.

## Evaluation using pretrained models

1. Download the pretrained checkpoints through the linkings in the above chart.

2. Testing

   To test, use:

   ```bash
   python tools/test.py projects/NeRF-Det/configs/nerfdet_res50_2x_low_res.py ${CHECKPOINT_PATH}
   ```

## Citation

<!-- You may remove this section if not applicable. -->

```latex
@inproceedings{
  xu2023nerfdet,
  title={NeRF-Det: Learning Geometry-Aware Volumetric Representation for Multi-View 3D Object Detection},
  author={Xu, Chenfeng and Wu, Bichen and Hou, Ji and Tsai, Sam and Li, Ruilong and Wang, Jialiang and Zhan, Wei and He, Zijian and Vajda, Peter and Keutzer, Kurt and Tomizuka, Masayoshi},
  booktitle={ICCV},
  year={2023},
}

@inproceedings{
park2023time,
title={Time Will Tell: New Outlooks and A Baseline for Temporal Multi-View 3D Object Detection},
author={Jinhyung Park and Chenfeng Xu and Shijia Yang and Kurt Keutzer and Kris M. Kitani and Masayoshi Tomizuka and Wei Zhan},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=H3HcEJA2Um}
}
```
