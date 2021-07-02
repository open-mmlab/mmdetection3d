## Changelog

### v0.15.0 (1/7/2021)

#### Compatibility

In order to fix the problem that the priority of EvalHook is too low, all hook priorities have been re-adjusted in 1.3.8, so MMDetection 2.14.0 needs to rely on the latest MMCV 1.3.8 version. For related information, please refer to [#1120](https://github.com/open-mmlab/mmcv/pull/1120), for related issues, please refer to [#5343](https://github.com/open-mmlab/mmdetection/issues/5343).

#### Highlights

- Support [PAConv](https://arxiv.org/abs/2103.14635)
- Support monocular/multi-view 3D detector [ImVoxelNet](https://arxiv.org/abs/2106.01178) on KITTI
- Support Transformer-based 3D detection method [Group-Free-3D](https://arxiv.org/abs/2104.00678) on ScanNet
- Add documentation for tasks including LiDAR-based 3D detection, vision-only 3D detection and point-based 3D semantic segmentation
- Add dataset documents like ScanNet

#### New Features

- Support Group-Free-3D on ScanNet (#539)
- Support PAConv modules (#598, #599)
- Support ImVoxelNet on KITTI (#627, #654)

#### Improvements

- Add unit tests for pipeline functions `LoadImageFromFileMono3D`, `ObjectNameFilter` and `ObjectRangeFilter` (#615)
- Enhance [IndoorPatchPointSample](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/transforms_3d.py) (#617)
- Refactor model initialization methods based MMCV (#622)
- Add Chinese docs (#629)
- Add documentation for LiDAR-based 3D detection (#642)
- Unify intrinsic and extrinsic matrices for all datasets (#653)
- Add documentation for point-based 3D semantic segmentation (#663)
- Add documentation of ScanNet for 3D detection (#664)
- Refine docs for tutorials (#666)
- Add documentation for vision-only 3D detection (#669)
- Refine docs for Quick Run and Useful Tools (#686)


#### Bug Fixes

- Fix the bug of [BackgroundPointsFilter](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/transforms_3d.py) using the bottom center of ground truth (#609)
- Fix [LoadMultiViewImageFromFiles](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/pipelines/loading.py) to unravel stacked multi-view images to list to be consistent with DefaultFormatBundle (#611)
- Fix the potential bug in [analyze_logs](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/analysis_tools/analyze_logs.py) when the training resumes from a checkpoint or is stopped before evaluation (#634)
- Fix test commands in docs and make some refinements (#635)
- Fix wrong config paths in unit tests (#641)


### v0.14.0 (1/6/2021)

#### Highlights
- Support the point cloud segmentation method [PointNet++](https://arxiv.org/abs/1706.02413)

#### New Features

- Support PointNet++ (#479, #528, #532, #541)
- Support RandomJitterPoints transform for point cloud segmentation (#584)
- Support RandomDropPointsColor transform for point cloud segmentation (#585)

#### Improvements

- Move the point alignment of ScanNet from data pre-processing to pipeline (#439, #470)
- Add compatibility document to provide detailed descriptions of BC-breaking changes (#504)
- Add MMSegmentation installation requirement (#535)
- Support points rotation even without bounding box in GlobalRotScaleTrans for point cloud segmentaiton (#540)
- Support visualization of detection results and dataset browse for nuScenes Mono-3D dataset (#542, #582)
- Support faster implementation of KNN (#586)
- Support RegNetX models on Lyft dataset (#589)
- Remove a useless parameter `label_weight` from segmentation datasets including `Custom3DSegDataset`, `ScanNetSegDataset` and `S3DISSegDataset` (#607)

#### Bug Fixes
- Fix a corrupted lidar data file in Lyft dataset in [data_preparation](https://github.com/open-mmlab/mmdetection3d/tree/master/docs/data_preparation.md) (#546)
- Fix evaluation bugs in nuScenes and Lyft dataset (#549)
- Fix converting points between coordinates with specific transformation matrix in the [coord_3d_mode.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/coord_3d_mode.py) (#556)
- Support PointPillars models on Lyft dataset (#578)
- Fix the bug of demo with pre-trained VoteNet model on ScanNet (#600)


### v0.13.0 (1/5/2021)

#### Highlights
- Support a monocular 3D detection method [FCOS3D](https://arxiv.org/abs/2104.10956)
- Support ScanNet and S3DIS semantic segmentation dataset
- Enhancement of visualization tools for dataset browsing and demos, including support of visualization for multi-modality data and point cloud segmentation.

#### New Features

- Support ScanNet semantic segmentation dataset (#390)
- Support monocular 3D detection on nuScenes (#392)
- Support multi-modality visualization (#405)
- Support nuimages visualization (#408)
- Support monocular 3D detection on KITTI (#415)
- Support online visualization of semantic segmentation results (#416)
- Support ScanNet test results submission to online benchmark (#418)
- Support S3DIS data pre-processing and dataset class (#433)
- Support FCOS3D (#436, #442, #482, #484)
- Support dataset browse for multiple types of datasets (#467)
- Adding paper-with-code (PWC) metafile for each model in the model zoo (#485)

#### Improvements

- Support dataset browsing for SUNRGBD, ScanNet or KITTI points and detection results (#367)
- Add the pipeline to load data using file client (#430)
- Support to customize the type of runner (#437)
- Make pipeline functions process points and masks simultaneously when sampling points (#444)
- Add waymo unit tests (#455)
- Split the visualization of projecting points onto image from that for only points (#480)
- Efficient implementation of PointSegClassMapping (#489)
- Use the new model registry from mmcv (#495)

#### Bug Fixes

- Fix Pytorch 1.8 Compilation issue in the [scatter_points_cuda.cu](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/src/scatter_points_cuda.cu) (#404)
- Fix [dynamic_scatter](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/src/scatter_points_cuda.cu) errors triggered by empty point input (#417)
- Fix the bug of missing points caused by using break incorrectly in the voxelization (#423)
- Fix the missing `coord_type` in the waymo dataset [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/waymoD5-3d-3class.py) (#441)
- Fix errors in four unittest functions of [configs](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py), [test_detectors.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tests/test_models/test_detectors.py), [test_heads.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tests/test_models/test_heads/test_heads.py) (#453)
- Fix 3DSSD training errors and simplify configs (#462)
- Clamp 3D votes projections to image boundaries in ImVoteNet (#463)
- Update out-of-date names of pipelines in the [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/benchmark/hv_pointpillars_secfpn_3x8_100e_det3d_kitti-3d-car.py) of pointpillars benchmark (#474)
- Fix the lack of a placeholder when unpacking RPN targets in the [h3d_bbox_head.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/models/roi_heads/bbox_heads/h3d_bbox_head.py) (#508)
- Fix the incorrect value of `K` when creating pickle files for SUN RGB-D (#511)

### v0.12.0 (1/4/2021)

#### Highlights

- Support a new multi-modality method [ImVoteNet](https://arxiv.org/abs/2001.10692).
- Support pytorch 1.7 and 1.8
- Refactor the structure of tools and [train.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/train.py)/[test.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/test.py)

#### New Features

- Support LiDAR-based semantic segmentation metrics (#332)
- Support [ImVoteNet](https://arxiv.org/abs/2001.10692) (#352, #384)
- Support the KNN GPU operation (#360, #371)

#### Improvements

- Add FAQ for common problems in the documentation (#333)
- Refactor the structure of tools (#339)
- Refactor [train.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/train.py) and [test.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/test.py) (#343)
- Support demo on nuScenes (#353)
- Add 3DSSD checkpoints (#359)
- Update the Bibtex of CenterPoint (#368)
- Add citation format and reference to other OpenMMLab projects in the README (#374)
- Upgrade the mmcv version requirements (#376)
- Add numba and numpy version requirements in FAQ (#379)
- Avoid unnecessary for-loop execution of vfe layer creation (#389)
- Update SUNRGBD dataset documentation to stress the requirements for training ImVoteNet (#391)
- Modify vote head to support 3DSSD (#396)

#### Bug Fixes

- Fix missing keys `coord_type` in database sampler config (#345)
- Rename H3DNet configs (#349)
- Fix CI by using ubuntu 18.04 in github workflow (#350)
- Add assertions to avoid 4-dim points being input to [points_in_boxes](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/roiaware_pool3d/points_in_boxes.py) (#357)
- Fix the SECOND results on Waymo in the corresponding [README](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) (#363)
- Fix the incorrect adopted pipeline when adding val to workflow (#370)
- Fix a potential bug when indices used in the backwarding in ThreeNN (#377)
- Fix a compilation error triggered by [scatter_points_cuda.cu](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/src/scatter_points_cuda.cu) in pytorch 1.7 (#393)

### v0.11.0 (1/3/2021)

#### Highlights

- Support more friendly visualization interfaces based on open3d
- Support a faster and more memory-efficient implementation of DynamicScatter
- Refactor unit tests and details of configs

#### New Features

- Support new visualization methods based on open3d (#284, #323)

#### Improvements

- Refactor unit tests (#303)
- Move the key `train_cfg` and `test_cfg` into the model configs (#307)
- Update [README](https://github.com/open-mmlab/mmdetection3d/blob/master/README.md/) with [Chinese version](https://github.com/open-mmlab/mmdetection3d/blob/master/README_zh-CN.md/) and [instructions for getting started](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/getting_started.md/). (#310, #316)
- Support a faster and more memory-efficient implementation of DynamicScatter (#318, #326)

#### Bug Fixes

- Fix an unsupported bias setting in the unit test for centerpoint head (#304)
- Fix errors due to typos in the centerpoint head (#308)
- Fix a minor bug in [points_in_boxes.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/roiaware_pool3d/points_in_boxes.py) when tensors are not in the same device. (#317)
- Fix warning of deprecated usages of nonzero during training with pytorch 1.6 (#330)

### v0.10.0 (1/2/2021)

#### Highlights

- Preliminary release of API for SemanticKITTI dataset.
- Documentation and demo enhancement for better user experience.
- Fix a number of underlying minor bugs and add some corresponding important unit tests.

#### New Features

- Support SemanticKITTI dataset preliminarily (#287)

#### Improvements

- Add tag to README in configurations for specifying different uses (#262)
- Update instructions for evaluation metrics in the documentation (#265)
- Add nuImages entry in [README.md](https://github.com/open-mmlab/mmdetection3d/blob/master/README.md/) and gif demo (#266, #268)
- Add unit test for voxelization (#275)

#### Bug Fixes

- Fixed the issue of unpacking size in [furthest_point_sample.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/furthest_point_sample/furthest_point_sample.py) (#248)
- Fix bugs for 3DSSD triggered by empty ground truths (#258)
- Remove models without checkpoints in model zoo statistics of documentation (#259)
- Fix some unclear installation instructions in [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/getting_started.md/) (#269)
- Fix relative paths/links in the documentation (#271)
- Fix a minor bug in [scatter_points_cuda.cu](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/src/scatter_points_cuda.cu) when num_features != 4 (#275)
- Fix the bug about missing text files when testing on KITTI (#278)
- Fix issues caused by inplace modification of tensors in `BaseInstance3DBoxes` (#283)
- Fix log analysis for evaluation and adjust the documentation accordingly (#285)

### v0.9.0 (31/12/2020)

#### Highlights

- Documentation refactoring with better structure, especially about how to implement new models and customized datasets.
- More compatible with refactored point structure by bug fixes in ground truth sampling.

#### Improvements

- Documentation refactoring (#242)

#### Bug Fixes

- Fix point structure related bugs in ground truth sampling (#211)
- Fix loading points in ground truth sampling augmentation on nuScenes (#221)
- Fix channel setting in the SeparateHead of CenterPoint (#228)
- Fix evaluation for indoors 3D detection in case of less classes in prediction (#231)
- Remove unreachable lines in nuScenes data converter (#235)
- Minor adjustments of numpy implementation for perspective projection and prediction filtering criterion in KITTI evaluation (#241)

### v0.8.0 (30/11/2020)

#### Highlights

- Refactor points structure with more constructive and clearer implementation.
- Support axis-aligned IoU loss for VoteNet with better performance.
- Update and enhance [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) benchmark on Waymo.

#### New Features

- Support axis-aligned IoU loss for VoteNet. (#194)
- Support points structure for consistent processing of all the point related representation. (#196, #204)

#### Improvements

- Enhance [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) benchmark on Waymo with stronger baselines. (#205)
- Add model zoo statistics and polish the documentation. (#201)

### v0.7.0 (1/11/2020)

#### Highlights

- Support a new method [SSN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700579.pdf) with benchmarks on nuScenes and Lyft datasets.
- Update benchmarks for SECOND on Waymo, CenterPoint with TTA on nuScenes and models with mixed precision training on KITTI and nuScenes.
- Support semantic segmentation on nuImages and provide [HTC](https://arxiv.org/abs/1901.07518) models with configurations and performance for reference.

#### New Features

- Modified primitive head which can support the setting on SUN-RGBD dataset (#136)
- Support semantic segmentation and [HTC](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/nuimages) with models for reference on nuImages dataset (#155)
- Support [SSN](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/ssn) on nuScenes and Lyft datasets (#147, #174, #166, #182)
- Support double flip for test time augmentation of CenterPoint with updated benchmark (#143)

#### Improvements

- Update [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) benchmark with configurations for reference on Waymo (#166)
- Delete checkpoints on Waymo to comply its specific license agreement (#180)
- Update models and instructions with [mixed precision training](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/fp16) on KITTI and nuScenes (#178)

#### Bug Fixes

- Fix incorrect code weights in anchor3d_head when introducing mixed precision training (#173)
- Fix the incorrect label mapping on nuImages dataset (#155)

### v0.6.1 (11/10/2020)

#### Highlights

- Support mixed precision training of voxel-based methods
- Support docker with pytorch 1.6.0
- Update baseline configs and results ([CenterPoint](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/centerpoint) on nuScenes and [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) on Waymo with full dataset)
- Switch model zoo to download.openmmlab.com

#### New Features

- Support dataset pipeline `VoxelBasedPointSampler` to sample multi-sweep points based on voxelization. (#125)
- Support mixed precision training of voxel-based methods (#132)
- Support docker with pytorch 1.6.0 (#160)

#### Improvements

- Reduce requirements for the case exclusive of Waymo (#121)
- Switch model zoo to download.openmmlab.com (#126)
- Update docs related to Waymo (#128)
- Add version assertion in the [init file](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/__init__.py) (#129)
- Add evaluation interval setting for CenterPoint (#131)
- Add unit test for CenterPoint (#133)
- Update [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) baselines on Waymo with full dataset (#142)
- Update [CenterPoint](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/centerpoint) results with models and logs (#154)

#### Bug Fixes

- Fix a bug of visualization in multi-batch case (#120)
- Fix bugs in dcn unit test (#130)
- Fix dcn bias bug in centerpoint (#137)
- Fix dataset mapping in the evaluation of nuScenes mini dataset (#140)
- Fix origin initialization in `CameraInstance3DBoxes` (#148, #150)
- Correct documentation link in the getting_started.md (#159)
- Fix model save path bug in gather_models.py (#153)
- Fix image padding shape bug in `PointFusion` (#162)

### v0.6.0 (20/9/2020)

#### Highlights

- Support new methods [H3DNet](https://arxiv.org/abs/2006.05682), [3DSSD](https://arxiv.org/abs/2002.10187), [CenterPoint](https://arxiv.org/abs/2006.11275).
- Support new dataset [Waymo](https://waymo.com/open/) (with PointPillars baselines) and [nuImages](https://www.nuscenes.org/nuimages) (with Mask R-CNN and Cascade Mask R-CNN baselines).
- Support Batch Inference
- Support Pytorch 1.6
- Start to publish `mmdet3d` package to PyPI since v0.5.0. You can use mmdet3d through `pip install mmdet3d`.

#### Backwards Incompatible Changes

- Support Batch Inference (#95, #103, #116): MMDetection3D v0.6.0 migrates to support batch inference based on MMDetection >= v2.4.0. This change influences all the test APIs in MMDetection3D and downstream codebases.
- Start to use collect environment function from MMCV (#113): MMDetection3D v0.6.0 migrates to use `collect_env` function in MMCV.
`get_compiler_version` and `get_compiling_cuda_version` compiled in `mmdet3d.ops.utils` are removed. Please import these two functions from `mmcv.ops`.

#### New Features

- Support [nuImages](https://www.nuscenes.org/nuimages) dataset by converting them into coco format and release Mask R-CNN and Cascade Mask R-CNN baseline models (#91, #94)
- Support to publish to PyPI in github-action (#17, #19, #25, #39, #40)
- Support CBGSDataset and make it generally applicable to all the supported datasets (#75, #94)
- Support [H3DNet](https://arxiv.org/abs/2006.05682) and release models on ScanNet dataset (#53, #58, #105)
- Support Fusion Point Sampling used in [3DSSD](https://arxiv.org/abs/2002.10187) (#66)
- Add `BackgroundPointsFilter` to filter background points in data pipeline (#84)
- Support pointnet2 with multi-scale grouping in backbone and refactor pointnets (#82)
- Support dilated ball query used in [3DSSD](https://arxiv.org/abs/2002.10187) (#96)
- Support [3DSSD](https://arxiv.org/abs/2002.10187) and release models on KITTI dataset (#83, #100, #104)
- Support [CenterPoint](https://arxiv.org/abs/2006.11275) and release models on nuScenes dataset (#49, #92)
- Support [Waymo](https://waymo.com/open/) dataset and release PointPillars baseline models (#118)
- Allow `LoadPointsFromMultiSweeps` to pad empty sweeps and select multiple sweeps randomly (#67)

#### Improvements

- Fix all warnings and bugs in Pytorch 1.6.0 (#70, #72)
- Update issue templates (#43)
- Update unit tests (#20, #24, #30)
- Update documentation for using `ply` format point cloud data (#41)
- Use points loader to load point cloud data in ground truth (GT) samplers (#87)
- Unify version file of OpenMMLab projects by using `version.py` (#112)
- Remove unnecessary data preprocessing commands of SUN RGB-D dataset (#110)

#### Bug Fixes

- Rename CosineAnealing to CosineAnnealing (#57)
- Fix device inconsistant bug in 3D IoU computation (#69)
- Fix a minor bug in json2csv of lyft dataset (#78)
- Add missed test data for pointnet modules (#85)
- Fix `use_valid_flag` bug in `CustomDataset` (#106)

### v0.5.0 (9/7/2020)

MMDetection3D is released.
