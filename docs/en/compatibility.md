## 0.16.0

### Returned values of `QueryAndGroup` operation

We modified the returned `grouped_xyz` value of operation `QueryAndGroup` to support PAConv segmentor. Originally, the `grouped_xyz` is centered by subtracting the grouping centers, which represents the relative positions of grouped points. Now, we didn't perform such subtraction and the returned `grouped_xyz` stands for the absolute coordinates of these points.

Note that, the other returned variables of `QueryAndGroup` such as `new_features`, `unique_cnt` and `grouped_idx` are not affected.

### NuScenes coco-style data pre-processing

We remove the rotation and dimension hack in the monocular 3D detection on nuScenes. Specifically, we transform the rotation and dimension of boxes defined by nuScenes devkit to the coordinate system of our `CameraInstance3DBoxes` in the pre-processing and transform them back in the post-processing. In this way, we can remove the corresponding [hack](https://github.com/open-mmlab/mmdetection3d/pull/744/files#diff-5bee5062bd84e6fa25a2fdd71353f6f283dfdc4a66a0316c3b1ca26078c978b6L165) used in the visualization tools. The modification also guarantees the correctness of all the operations based on our `CameraInstance3DBoxes` (such as NMS and flip augmentation) when training monocular 3D detectors.

The modification only influences nuScenes coco-style json files. Please re-run the nuScenes data preparation script if necessary. See more details in the PR [#744](https://github.com/open-mmlab/mmdetection3d/pull/744).

### ScanNet dataset for ImVoxelNet

We adopt a new pre-processing procedure for the ScanNet dataset in order to support ImVoxelNet, which is a multi-view method requiring image data. In previous versions of MMDetection3D, ScanNet dataset was only used for point cloud based 3D detection and segmentation methods. We plan adding ImVoxelNet to our model zoo, thus updating ScanNet correspondingly by adding image-related pre-processing steps. Specifically, we made these changes:

- Add [script](https://github.com/open-mmlab/mmdetection3d/blob/master/data/scannet/extract_posed_images.py) for extracting RGB data.
- Update [script](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/data_converter/scannet_data_utils.py) for annotation creating.
- Add instructions in the documents on preparing image data.

Please refer to the ScanNet [README.md](https://github.com/open-mmlab/mmdetection3d/blob/master/data/scannet/README.md/) for more details.

## 0.15.0

### MMCV Version

In order to fix the problem that the priority of EvalHook is too low, all hook priorities have been re-adjusted in 1.3.8, so MMDetection 2.14.0 needs to rely on the latest MMCV 1.3.8 version. For related information, please refer to [#1120](https://github.com/open-mmlab/mmcv/pull/1120), for related issues, please refer to [#5343](https://github.com/open-mmlab/mmdetection/issues/5343).

### Unified parameter initialization

To unify the parameter initialization in OpenMMLab projects, MMCV supports `BaseModule` that accepts `init_cfg` to allow the modules' parameters initialized in a flexible and unified manner. Now the users need to explicitly call `model.init_weights()` in the training script to initialize the model (as in [here](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/train.py#L183), previously this was handled by the detector. Please refer to PR [#622](https://github.com/open-mmlab/mmdetection3d/pull/622) for details.

### BackgroundPointsFilter

We modified the dataset augmentation function `BackgroundPointsFilter`([here](https://github.com/open-mmlab/mmdetection3d/blob/v0.15.0/mmdet3d/datasets/pipelines/transforms_3d.py#L1132)). In previous version of MMdetection3D, `BackgroundPointsFilter` changes the gt_bboxes_3d's bottom center to the gravity center. In MMDetection3D 0.15.0,
`BackgroundPointsFilter` will not change it. Please refer to PR [#609](https://github.com/open-mmlab/mmdetection3d/pull/609) for details.

### Enhance `IndoorPatchPointSample` transform

We enhance the pipeline function `IndoorPatchPointSample` used in point cloud segmentation task by adding more choices for patch selection. Also, we plan to remove the unused parameter `sample_rate` in the future. Please modify the code as well as the config files accordingly if you use this transform.

## 0.14.0

### Dataset class for 3D segmentation task

We remove a useless parameter `label_weight` from segmentation datasets including `Custom3DSegDataset`, `ScanNetSegDataset` and `S3DISSegDataset` since this weight is utilized in the loss function of model class. Please modify the code as well as the config files accordingly if you use or inherit from these codes.

### ScanNet data pre-processing

We adopt new pre-processing and conversion steps of ScanNet dataset. In previous versions of MMDetection3D, ScanNet dataset was only used for 3D detection task, where we trained on the training set and tested on the validation set. In MMDetection3D 0.14.0, we further support 3D segmentation task on ScanNet, which includes online benchmarking on test set. Since the alignment matrix is not provided for test set data, we abandon the alignment of points in data generation steps to support both tasks. Besides, as 3D segmentation requires per-point prediction, we also remove the down-sampling step in data generation.

- In the new ScanNet processing scripts, we save the unaligned points for all the training, validation and test set. For train and val set with annotations, we also store the `axis_align_matrix` in data infos. For ground-truth bounding boxes, we store boxes in both aligned and unaligned coordinates with key `gt_boxes_upright_depth` and key `unaligned_gt_boxes_upright_depth` respectively in data infos.

- In `ScanNetDataset`, we now load the `axis_align_matrix` as a part of data annotations. If it is not contained in old data infos, we will use identity matrix for compatibility. We also add a transform function `GlobalAlignment` in ScanNet detection data pipeline to align the points.

- Since the aligned boxes share the same key as in old data infos, we do not need to modify the code related to it. But do remember that they are not in the same coordinate system as the saved points.

- There is an `PointSample` pipeline in the data pipelines for ScanNet detection task which down-samples points. So removing down-sampling in data generation will not affect the code.

We have trained a [VoteNet](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/votenet/votenet_8x8_scannet-3d-18class.py) model on the newly processed ScanNet dataset and get similar benchmark results. In order to prepare ScanNet data for both detection and segmentation tasks, please re-run the new pre-processing scripts following the ScanNet [README.md](https://github.com/open-mmlab/mmdetection3d/blob/master/data/scannet/README.md/).

## 0.12.0

### SUNRGBD dataset for ImVoteNet

We adopt a new pre-processing procedure for the SUNRGBD dataset in order to support ImVoteNet, which is a multi-modality method requiring both image and point cloud data. In previous versions of MMDetection3D, SUNRGBD dataset was only used for point cloud based 3D detection methods. In MMDetection3D 0.12.0, we add ImVoteNet to our model zoo, thus updating SUNRGBD correspondingly by adding image-related pre-processing steps. Specifically, we made these changes:

- Fix a bug in the image file path in meta data.
- Convert calibration matrices from double to float to avoid type mismatch in further operations.
- Add instructions in the documents on preparing image data.

Please refer to the SUNRGBD [README.md](https://github.com/open-mmlab/mmdetection3d/blob/master/data/sunrgbd/README.md/) for more details.

## 0.6.0

### VoteNet and H3DNet model structure update

In MMDetection 0.6.0, we updated the model structures of VoteNet and H3DNet, therefore model checkpoints generated by MMDetection < 0.6.0 should be first converted to a format compatible with the latest structures via [convert_votenet_checkpoints.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/model_converters/convert_votenet_checkpoints.py) and [convert_h3dnet_checkpoints.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/model_converters/convert_h3dnet_checkpoints.py) . For more details, please refer to the VoteNet [README.md](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/votenet/README.md/) and H3DNet [README.md](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/h3dnet/README.md/).
