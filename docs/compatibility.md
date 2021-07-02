# Compatibility with Previous Versions of MMDetection3D

This document provides detailed descriptions of the BC-breaking changes in MMDetection3D.

## MMDetection3D 0.15.0

### MMCV Version

In order to fix the problem that the priority of EvalHook is too low, all hook priorities have been re-adjusted in 1.3.8, so MMDetection 2.14.0 needs to rely on the latest MMCV 1.3.8 version. For related information, please refer to [#1120](https://github.com/open-mmlab/mmcv/pull/1120), for related issues, please refer to [#5343](https://github.com/open-mmlab/mmdetection/issues/5343).

### Unified parameter initialization

To unify the parameter initialization in OpenMMLab projects, MMCV supports `BaseModule` that accepts `init_cfg` to allow the modules' parameters initialized in a flexible and unified manner. Now the users need to explicitly call `model.init_weights()` in the training script to initialize the model (as in [here](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/train.py#L183), previously this was handled by the detector. Please refer to PR #622 for details.

### BackgroundPointsFilter

We modified the dataset aumentation function `BackgroundPointsFilter`(in [here](https://github.com/open-mmlab/mmdetection3d/blob/mmdet3d/datasets/pipelines/transforms_3d.py#L1101)). In previous version of MMdetection3D, `BackgroundPointsFilter` changes the gt_bboxes_3d's bottom center to the gravity center. In MMDetection3D 0.15.0,
`BackgroundPointsFilter` will not change it. Please refer to PR #609 for details.

### Enhance `IndoorPatchPointSample` transform

We enhance the pipeline function `IndoorPatchPointSample` used in point cloud segmentation task by adding more choices for patch selection. Also, we plan to remove the unused parameter `sample_rate` in the future. Please modify the code as well as the config files accordingly if you use this transform.

## MMDetection3D 0.14.0

### Dataset class for 3D segmentation task

We remove a useless parameter `label_weight` from segmentation datasets including `Custom3DSegDataset`, `ScanNetSegDataset` and `S3DISSegDataset` since this weight is utilized in the loss function of model class. Please modify the code as well as the config files accordingly if you use or inherit from these codes.

### ScanNet data pre-processing

We adopt new pre-processing and conversion steps of ScanNet dataset. In previous versions of MMDetection3D, ScanNet dataset was only used for 3D detection task, where we trained on the training set and tested on the validation set. In MMDetection3D 0.14.0, we further support 3D segmentation task on ScanNet, which includes online benchmarking on test set. Since the alignment matrix is not provided for test set data, we abandon the alignment of points in data generation steps to support both tasks. Besides, as 3D segmentation requires per-point prediction, we also remove the down-sampling step in data generation.

- In the new ScanNet processing scripts, we save the unaligned points for all the training, validation and test set. For train and val set with annotations, we also store the `axis_align_matrix` in data infos. For ground-truth bounding boxes, we store boxes in both aligned and unaligned coordinates with key `gt_boxes_upright_depth` and key `unaligned_gt_boxes_upright_depth` respectively in data infos.

- In `ScanNetDataset`, we now load the `axis_align_matrix` as a part of data annotations. If it is not contained in old data infos, we will use identity matrix for compatibility. We also add a transform function `GlobalAlignment` in ScanNet detection data pipeline to align the points.

- Since the aligned boxes share the same key as in old data infos, we do not need to modify the code related to it. But do remember that they are not in the same coordinate system as the saved points.

- There is an `IndoorPointSample` pipeline in the data pipelines for ScanNet detection task which down-samples points. So removing down-sampling in data generation will not affect the code.

We have trained a [VoteNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/votenet/votenet_8x8_scannet-3d-18class.py) model on the newly processed ScanNet dataset and get similar benchmark results. In order to prepare ScanNet data for both detection and segmentation tasks, please re-run the new pre-processing scripts following the ScanNet [README.md](https://github.com/open-mmlab/mmdetection3d/blob/master/data/scannet/README.md/).

## MMDetection3D 0.12.0

### SUNRGBD dataset for ImVoteNet

We adopt a new pre-processing procedure for the SUNRGBD dataset in order to support ImVoteNet, which is a multi-modality method requiring both image and point cloud data. In previous versions of MMDetection3D, SUNRGBD dataset was only used for point cloud based 3D detection methods. In MMDetection3D 0.12.0, we add ImVoteNet to our model zoo, thus updating SUNRGBD correspondingly by adding image-related pre-processing steps. Specificly, we made these changes:

- Fix a bug in the image file path in meta data.
- Convert calibration matrices from double to float to avoid type mismatch in further operations.
- Add instructions in the documents on preparing image data.

Please refer to the SUNRGBD [README.md](https://github.com/open-mmlab/mmdetection3d/blob/master/data/sunrgbd/README.md/) for more details.

## MMDetection3D 0.6.0

### VoteNet model structure update

In MMDetection 0.6.0, we updated the model structure of VoteNet, therefore model checkpoints generated by MMDetection < 0.6.0 should be first converted to a format compatible with the latest VoteNet structure via this [script](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/model_converters/convert_votenet_checkpoints.py). For more details, please refer to the VoteNet [README.md](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/votenet/README.md/)
