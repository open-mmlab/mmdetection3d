# NuImages Results

## Introduction

We support and provide some baseline results on [nuImages dataset](https://www.nuscenes.org/nuimages).
We follow the class mapping in nuScenes dataset, which maps the original categories into 10 foreground categories.
The baseline results include instance segmentation models, e.g., Mask R-CNN and Cascade Mask R-CNN.
We will support panoptic segmentation models in the future.

The dataset converted by the script of v0.6.0 only supports instance segmentation. Since v0.7.0, we also support to produce semantic segmentation mask of each image; thus, we can train HTC or semantic segmentation models using the dataset. To convert the nuImages dataset into COCO format, please use the command below:

```shell
python -u tools/data_converter/nuimage_converter.py --data-root ${DATA_ROOT} --version ${VERIONS} \
                                                    --out-dir ${OUT_DIR} --nproc ${NUM_WORKERS} --extra-tag ${TAG}
```

- `--data-root`: the root of the dataset, defaults to `./data/nuimages`.
- `--version`: the version of the dataset, defaults to `v1.0-mini`. To get the full dataset, please use `--version v1.0-train v1.0-val v1.0-mini`
- `--out-dir`: the output directory of annotations and semantic masks, defaults to `./data/nuimages/annotations/`.
- `--nproc`: number of workers for data preparation, defaults to `4`. Larger number could reduce the preparation time as images are processed in parallel.
- `--extra-tag`: extra tag of the annotations, defaults to `nuimages`. This can be used to separate different annotations processed in different time for study.

## Results

### Instance Segmentation

We report Mask R-CNN and Cascade Mask R-CNN results on nuimages.

|Method | Backbone|Pretraining | Lr schd | Mem (GB) | Box AP  | Mask AP  |Download |
| :---------: |:---------: | :---------: | :-----: |:-----: | :------: | :------------: | :----: |
| Mask R-CNN| [R-50](./mask_rcnn_r50_fpn_1x_nuim.py) |IN|1x|7.4|47.8 |38.4|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_fpn_1x_nuim/mask_rcnn_r50_fpn_1x_nuim_20201008_195238-e99f5182.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_fpn_1x_nuim/mask_rcnn_r50_fpn_1x_nuim_20201008_195238.log.json)|
| Mask R-CNN| [R-50](./mask_rcnn_r50_fpn_coco-2x_1x_nuim.py) |IN+COCO-2x|1x|7.4|49.7|40.5|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_fpn_coco-2x_1x_nuim/mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_fpn_coco-2x_1x_nuim/mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238.log.json)|
| Mask R-CNN| [R-50-CAFFE](./mask_rcnn_r50_caffe_fpn_1x_nuim.py) |IN|1x|7.0|47.7|38.2|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_caffe_fpn_1x_nuim/) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_caffe_fpn_1x_nuim/)|
| Mask R-CNN| [R-50-CAFFE](./mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim.py) |IN+COCO-3x|1x|7.0|49.9|40.8|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim/mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim_20201008_195305-661a992e.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim/mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim_20201008_195305.log.json)|
| Mask R-CNN| [R-50-CAFFE](./mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim.py) |IN+COCO-3x|20e|7.0|50.6|41.3|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_caffe_fpn_coco-3x_20e_nuim/mask_rcnn_r50_caffe_fpn_coco-3x_20e_nuim_20201009_125002-5529442c.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_caffe_fpn_coco-3x_20e_nuim/mask_rcnn_r50_caffe_fpn_coco-3x_20e_nuim_20201009_125002.log.json)|
| Mask R-CNN| [R-101](./mask_rcnn_r101_fpn_1x_nuim.py) |IN|1x|10.9|48.9|39.1|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r101_fpn_1x_nuim/mask_rcnn_r101_fpn_1x_nuim_20201024_134803-65c7623a.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r101_fpn_1x_nuim/mask_rcnn_r101_fpn_1x_nuim_20201024_134803.log.json)|
| Mask R-CNN| [X-101_32x4d](./mask_rcnn_x101_32x4d_fpn_1x_nuim.py) |IN|1x|13.3|50.4|40.5|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_x101_32x4d_fpn_1x_nuim/mask_rcnn_x101_32x4d_fpn_1x_nuim_20201024_135741-b699ab37.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_x101_32x4d_fpn_1x_nuim/mask_rcnn_x101_32x4d_fpn_1x_nuim_20201024_135741.log.json)|
| Cascade Mask R-CNN| [R-50](./cascade_mask_rcnn_r50_fpn_1x_nuim.py) |IN|1x|8.9|50.8|40.4|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_1x_nuim/cascade_mask_rcnn_r50_fpn_1x_nuim_20201008_195342-1147c036.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_1x_nuim/cascade_mask_rcnn_r50_fpn_1x_nuim_20201008_195342.log.json)|
| Cascade Mask R-CNN| [R-50](./cascade_mask_rcnn_r50_fpn_coco-20e_1x_nuim.py) |IN+COCO-20e|1x|8.9|52.8|42.2|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_1x_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_1x_nuim_20201009_124158-ad0540e3.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_1x_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_1x_nuim_20201009_124158.log.json)|
| Cascade Mask R-CNN| [R-50](./cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim.py) |IN+COCO-20e|20e|8.9|52.8|42.2|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951.log.json)|
| Cascade Mask R-CNN| [R-101](./cascade_mask_rcnn_r101_fpn_1x_nuim.py) |IN|1x|12.5|51.5|40.7|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r101_fpn_1x_nuim/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r101_fpn_1x_nuim/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804.log.json)|
| Cascade Mask R-CNN| [X-101_32x4d](./cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim.py) |IN|1x|14.9|52.8|41.6|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim/cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim_20201024_135753-e0e49778.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim/cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim_20201024_135753.log.json)|
| HTC w/o semantic|[R-50](./htc_without_semantic_r50_fpn_1x_nuim.py) |IN|1x||[model]() &#124; [log]()|
| HTC|[R-50](./htc_r50_fpn_1x_nuim.py) |IN|1x||[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/)|
| HTC|[R-50](./htc_r50_fpn_coco-20e_1x_nuim.py) |IN+COCO-20e|1x|11.6|53.8|43.8|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/htc_r50_fpn_coco-20e_1x_nuim/htc_r50_fpn_coco-20e_1x_nuim_20201010_070203-0b53a65e.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/htc_r50_fpn_coco-20e_1x_nuim/htc_r50_fpn_coco-20e_1x_nuim_20201010_070203.log.json)|
| HTC|[R-50](./htc_r50_fpn_coco-20e_20e_nuim.py) |IN+COCO-20e|20e|11.6|54.8|44.4|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/htc_r50_fpn_coco-20e_20e_nuim/htc_r50_fpn_coco-20e_20e_nuim_20201008_211415-d6c60a2c.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/htc_r50_fpn_coco-20e_20e_nuim/htc_r50_fpn_coco-20e_20e_nuim_20201008_211415.log.json)|
| HTC|[X-101_64x4d + DCN_c3-c5](./htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim.py) |IN+COCO-20e|20e|13.3|57.3|46.4|[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim_20201008_211222-0b16ac4b.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim_20201008_211222.log.json)|

**Note**:
1. `IN` means only using ImageNet pre-trained backbone. `IN+COCO-Nx` and `IN+COCO-Ne` means the backbone is first pre-trained on ImageNet, and then the detector is pre-trained on COCO train2017 dataset by `Nx` and `N` epochs schedules, respectively.
2. All the training hyper-parameters follow the standard schedules on COCO dataset except that the images are resized from
1280 x 720 to 1920 x 1080 (relative ratio 0.8 to 1.2) since the images are in size 1600 x 900.
3. The class order in the detectors released in v0.6.0 is different from the order in the configs because the bug in the convertion script. This bug has been fixed since v0.7.0 and the models trained by the correct class order are also released. If you used nuImages since v0.6.0, please re-convert the data through the convertion script using the above-mentioned command.
