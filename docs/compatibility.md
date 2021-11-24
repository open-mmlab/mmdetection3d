## v1.0.0.dev0

### Coordinate system refactoring

In this version, we did a major code refactoring which improved the consistency among the three coordinate systems (and corresponding box representation), LiDAR, Camera, and Depth. A brief summary for this refactoring is as follows:

- The three coordinate systems are all right-handed now (which means the yaw angle increases in the counterclockwise direction).
- The LiDAR system `(x_size, y_size, z_size)` corresponds to `(l, w, h)` instead of `(w, l, h)`. This is more natural since `l` is parallel with the direction where the yaw angle is zero, and we prefer using the positive direction of the `x` axis as that direction, which is exactly how we define yaw angle in Depth and Camera coordinate systems.
- The APIs for box-related operations are improved and now are more user-friendly.

#### ***NOTICE!!***

Since definitions of box representation have changed, the annotation data of most datasets require updating:
- SUN RGB-D: Yaw angles in the annotation should be reversed.
- KITTI: For LiDAR boxes in GT databases, (x_size, y_size, z_size, yaw) out of (x, y, z, x_size, y_size, z_size) should be converted from the old LiDAR coordinate system to the new one. The training/validation data annotations should be left unchanged since they are under the Camera coordinate system, which is unmodified after the refactoring.
- Waymo: Same as KITTI.
- nuScenes: For LiDAR boxes in training/validation data and GT databases, (x_size, y_size, z_size, yaw) out of (x, y, z, x_size, y_size, z_size) should be converted.
- Lyft: Same as nuScenes.

Please regenerate the data annotation/GT database files or use [`update_data_coords.py`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/tools/update_data_coords.py) to update the data.

To use boxes under Depth and LiDAR coordinate systems, or to convert boxes between different coordinate systems, users should be aware of the difference between the old and new definitions. For example, the rotation, flipping, and bev functions of [`DepthInstance3DBoxes`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/bbox/structures/depth_box3d.py) and [`LiDARInstance3DBoxes`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mdet3d/core/bbox/structures/lidar_box3d.py) and box conversion [functions](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/bbox/structures/box_3d_mode.py) have all been reimplemented in the refactoring.

Consequently, functions like [`output_to_lyft_box`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/datasets/lyft_dataset.py) undergo small modification to adapt to the new LiDAR/Depth box.

Since the LiDAR system `(x_size, y_size, z_size)` now corresponds to `(l, w, h)` instead of `(w, l, h)`, the anchor sizes for LiDAR boxes are also changed, e.g., from `[1.6, 3.9, 1.56]` to `[3.9, 1.6, 1.56]`.

Functions only involving points are generally unaffected except if they rely on some refactored utility functions such as `rotation_3d_in_axis`.

#### Other BC-breaking or new features:

- `array_converter`: Please refer to [array_converter.py](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/utils/array_converter.py). Functions wrapped with `array_converter` can convert array-like input types of `torch.Tensor`, `np.ndarray`, and `list/tuple/float` to `torch.Tensor` to process in an unified PyTorch pipeline. The result may finally be converted back to the input type. Most functions in [utils.py](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/bbox/structures/utils.py) are wrapped with `array_converter`.
- [`points_in_boxes`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/bbox/structures/base_box3d.py) and [`points_in_boxes_batch`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/bbox/structures/base_box3d.py) will be deprecated soon. They are renamed to `points_in_boxes_part` and `points_in_boxes_all` respectively, with more detailed docstrings. The major difference of the two functions is that if a point is enclosed by multiple boxes, `points_in_boxes_part` will only return the index of the first enclosing box while `points_in_boxes_all` will return all the indices of enclosing boxes.
- `rotation_3d_in_axis`: Please refer to [utils.py](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/bbox/structures/utils.py). Now this function supports multiple input types and more options. The function with the same name in [box_np_ops.py](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/bbox/box_np_ops.py) is deleted since we do not need another function to tackle with NumPy data. `rotation_2d`, `points_cam2img`, and `limit_period` in box_np_ops.py are also deleted for the same reason.
- `bev` method of [`CameraInstance3DBoxes`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/core/bbox/structures/cam_box3d.py): Changed it to be consistent with the definition of bev in Depth and LiDAR coordinate systems.
- Data augmentation utils in [data_augment_utils.py](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/datasets/pipelines/data_augment_utils.py) now follow the rules of a right-handed system.
- We do not need the yaw hacking in KITTI anymore after refining [`get_direction_target`](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/mmdet3d/models/dense_heads/train_mixins.py). Interested users may refer to PR [#677](https://github.com/open-mmlab/mmdetection3d/pull/677) .


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
