## Changelog

### v1.0.0rc6 (2/12/2022)

#### New Features

- Add `Projects/` folder and the first example project (#2082)

#### Improvements

- Update Waymo converter to save storage space (#1759)
- Update model link and performance of CenterPoint (#1916)

#### Bug Fixes

- Fix GPU memory occupancy problem in PointRCNN (#1928)
- Fix sampling bug in `IoUNegPiecewiseSampler` (#2018)

#### Contributors

A total of 6 developers contributed to this release.

@oyel, @zzj403, @VVsssssk, @Tai-Wang, @tpoisonooo, @JingweiZhang12, @ZCMax

### v1.0.0rc5 (11/10/2022)

#### New Features

- Support ImVoxelNet on SUN RGB-D (#1738)

#### Improvements

- Fix the cross-codebase reference problem in metafile README (#1644)
- Update the Chinese documentation about getting started (#1715)
- Fix docs link and add docs link checker (#1811)

#### Bug Fixes

- Fix a visualization bug that is potentially triggered by empty prediction labels (#1725)
- Fix point cloud segmentation visualization bug due to wrong parameter passing (#1858)
- Fix Nan loss bug during PointRCNN training (#1874)

#### Contributors

A total of 11 developers contributed to this release.

@ZwwWayne, @Tai-Wang, @filaPro, @VVsssssk, @ZCMax, @Xiangxu-0103, @holtvogt, @tpoisonooo, @lianqing01, @TommyZihao, @aditya9710

### v1.0.0rc4 (8/8/2022)

#### Highlights

- Support [FCAF3D](https://arxiv.org/pdf/2112.00322.pdf)

#### New Features

- Support [FCAF3D](https://arxiv.org/pdf/2112.00322.pdf) (#1547)
- Add the transformation to support multi-camera 3D object detection (#1580)
- Support lift-splat-shoot view transformer (#1598)

#### Improvements

- Remove the limitation of the maximum number of points during SUN RGB-D preprocessing (#1555)
- Support circle CI (#1647)
- Add mim to extras_require in setup.py (#1560, #1574)
- Update dockerfile package version (#1697)

#### Bug Fixes

- Flip yaw angle for DepthInstance3DBoxes.overlaps (#1548, #1556)
- Fix DGCNN configs (#1587)
- Fix bbox head not registered bug (#1625)
- Fix missing objects in S3DIS preprocessing (#1665)
- Fix spconv2.0 model loading bug (#1699)

#### Contributors

A total of 9 developers contributed to this release.

@Tai-Wang, @ZwwWayne, @filaPro, @lianqing11, @ZCMax, @HuangJunJie2017, @Xiangxu-0103, @ChonghaoSima, @VVsssssk

### v1.0.0rc3 (8/6/2022)

#### Highlights

- Support [SA-SSD](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.pdf)

#### New Features

- Support [SA-SSD](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.pdf) (#1337)

#### Improvements

- Add Chinese documentation for vision-only 3D detection (#1438)
- Update CenterPoint pretrained models that are compatible with refactored coordinate systems (#1450)
- Configure myst-parser to parse anchor tag in the documentation (#1488)
- Replace markdownlint with mdformat for avoiding installing ruby (#1489)
- Add missing `gt_names` when getting annotation info in Custom3DDataset (#1519)
- Support S3DIS full ceph training (#1542)
- Rewrite the installation and FAQ documentation (#1545)

#### Bug Fixes

- Fix the incorrect registry name when building RoI extractors (#1460)
- Fix the potential problems caused by the registry scope update when composing pipelines (#1466) and using CocoDataset (#1536)
- Fix the missing selection with `order` in the [box3d_nms](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/post_processing/box3d_nms.py) introduced by [#1403](https://github.com/open-mmlab/mmdetection3d/pull/1403) (#1479)
- Update the [PointPillars config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py) to make it consistent with the log (#1486)
- Fix heading anchor in documentation (#1490)
- Fix the compatibility of mmcv in the dockerfile (#1508)
- Make overwrite_spconv packaged when building whl (#1516)
- Fix the requirement of mmcv and mmdet (#1537)
- Update configs of PartA2 and support its compatibility with spconv 2.0 (#1538)

#### Contributors

A total of 13 developers contributed to this release.

@Xiangxu-0103, @ZCMax, @jshilong, @filaPro, @atinfinity, @Tai-Wang, @wenbo-yu, @yi-chen-isuzu, @ZwwWayne, @wchen61, @VVsssssk, @AlexPasqua, @lianqing11

### v1.0.0rc2 (1/5/2022)

#### Highlights

- Support spconv 2.0
- Support MinkowskiEngine with MinkResNet
- Support training models on custom datasets with only point clouds
- Update Registry to distinguish the scope of built functions
- Replace mmcv.iou3d with a set of bird-eye-view (BEV) operators to unify the operations of rotated boxes

#### New Features

- Add loader arguments in the configuration files (#1388)
- Support [spconv 2.0](https://github.com/traveller59/spconv) when the package is installed. Users can still use spconv 1.x in MMCV with CUDA 9.0 (only cost more memory) without losing the compatibility of model weights between two versions (#1421)
- Support MinkowskiEngine with MinkResNet (#1422)

#### Improvements

- Add the documentation for model deployment (#1373, #1436)
- Add Chinese documentation of
  - Speed benchmark (#1379)
  - LiDAR-based 3D detection (#1368)
  - LiDAR 3D segmentation (#1420)
  - Coordinate system refactoring (#1384)
- Support training models on custom datasets with only point clouds (#1393)
- Replace mmcv.iou3d with a set of bird-eye-view (BEV) operators to unify the operations of rotated boxes (#1403, #1418)
- Update Registry to distinguish the scope of building functions (#1412, #1443)
- Replace recommonmark with myst_parser for documentation rendering (#1414)

#### Bug Fixes

- Fix the show pipeline in the [browse_dataset.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/misc/browse_dataset.py) (#1376)
- Fix missing __init__ files after coordinate system refactoring (#1383)
- Fix the incorrect yaw in the visualization caused by coordinate system refactoring (#1407)
- Fix `NaiveSyncBatchNorm1d` and `NaiveSyncBatchNorm2d` to support non-distributed cases and more general inputs (#1435)

#### Contributors

A total of 11 developers contributed to this release.

@ZCMax, @ZwwWayne, @Tai-Wang, @VVsssssk, @HanaRo, @JoeyforJoy, @ansonlcy, @filaPro, @jshilong, @Xiangxu-0103, @deleomike

### v1.0.0rc1 (1/4/2022)

#### Compatibility

- We migrate all the mmdet3d ops to mmcv and do not need to compile them when installing mmdet3d.
- To fix the imprecise timestamp and optimize its saving method, we reformat the point cloud data during Waymo data conversion. The data conversion time is also optimized significantly by supporting parallel processing. Please re-generate KITTI format Waymo data if necessary. See more details in the [compatibility documentation](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/compatibility.md).
- We update some of the model checkpoints after the refactor of coordinate systems. Please stay tuned for the release of the remaining model checkpoints.

|               | Fully Updated | Partially Updated | In Progress | No Influcence |
| ------------- | :-----------: | :---------------: | :---------: | :-----------: |
| SECOND        |               |         ✓         |             |               |
| PointPillars  |               |         ✓         |             |               |
| FreeAnchor    |       ✓       |                   |             |               |
| VoteNet       |       ✓       |                   |             |               |
| H3DNet        |       ✓       |                   |             |               |
| 3DSSD         |               |         ✓         |             |               |
| Part-A2       |       ✓       |                   |             |               |
| MVXNet        |       ✓       |                   |             |               |
| CenterPoint   |               |                   |      ✓      |               |
| SSN           |       ✓       |                   |             |               |
| ImVoteNet     |       ✓       |                   |             |               |
| FCOS3D        |               |                   |             |       ✓       |
| PointNet++    |               |                   |             |       ✓       |
| Group-Free-3D |               |                   |             |       ✓       |
| ImVoxelNet    |       ✓       |                   |             |               |
| PAConv        |               |                   |             |       ✓       |
| DGCNN         |               |                   |             |       ✓       |
| SMOKE         |               |                   |             |       ✓       |
| PGD           |               |                   |             |       ✓       |
| MonoFlex      |               |                   |             |       ✓       |

#### Highlights

- Migrate all the mmdet3d ops to mmcv
- Support parallel waymo data converter
- Add ScanNet instance segmentation dataset with metrics
- Better compatibility for windows with CI support, op migration and bug fixes
- Support loading annotations from Ceph

#### New Features

- Add ScanNet instance segmentation dataset with metrics (#1230)
- Support different random seeds for different ranks (#1321)
- Support loading annotations from Ceph (#1325)
- Support resuming from the latest checkpoint automatically (#1329)
- Add windows CI (#1345)

#### Improvements

- Update the table format and OpenMMLab project orders in [README.md](https://github.com/open-mmlab/mmdetection3d/blob/master/README.md) (#1272, #1283)
- Migrate all the mmdet3d ops to mmcv (#1240, #1286, #1290, #1333)
- Add `with_plane` flag in the KITTI data conversion (#1278)
- Update instructions and links in the documentation (#1300, 1309, #1319)
- Support parallel Waymo dataset converter and ground truth database generator (#1327)
- Add quick installation commands to [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md) (#1366)

#### Bug Fixes

- Update nuimages configs to use new nms config style (#1258)
- Fix the usage of np.long for windows compatibility (#1270)
- Fix the incorrect indexing in `BasePoints` (#1274)
- Fix the incorrect indexing in the [pillar_scatter.forward_single](https://github.com/open-mmlab/mmdetection3d/blob/dev/mmdet3d/models/middle_encoders/pillar_scatter.py#L38) (#1280)
- Fix unit tests that use GPUs (#1301)
- Fix incorrect feature dimensions in `DynamicPillarFeatureNet` caused by previous upgrading of `PillarFeatureNet` (#1302)
- Remove the `CameraPoints` constraint in `PointSample` (#1314)
- Fix imprecise timestamps saving of Waymo dataset (#1327)

#### Contributors

A total of 9 developers contributed to this release.

@ZCMax, @ZwwWayne, @wHao-Wu, @Tai-Wang, @wangruohui, @zjwzcx, @Xiangxu-0103, @EdAyers, @hongye-dev, @zhanggefan

### v1.0.0rc0 (18/2/2022)

#### Compatibility

- We refactor our three coordinate systems to make their rotation directions and origins more consistent, and further remove unnecessary hacks in different datasets and models. Therefore, please re-generate data infos or convert the old version to the new one with our provided scripts. We will also provide updated checkpoints in the next version. Please refer to the [compatibility documentation](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/docs/en/compatibility.md) for more details.
- Unify the camera keys for consistent transformation between coordinate systems on different datasets. The modification changes the key names to `lidar2img`, `depth2img`, `cam2img`, etc., for easier understanding. Customized codes using legacy keys may be influenced.
- The next release will begin to move files of CUDA ops to [MMCV](https://github.com/open-mmlab/mmcv). It will influence the way to import related functions. We will not break the compatibility but will raise a warning first and please prepare to migrate it.

#### Highlights

- Support new monocular 3D detectors: [PGD](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/pgd), [SMOKE](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/smoke), [MonoFlex](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/monoflex)
- Support a new LiDAR-based detector: [PointRCNN](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/point_rcnn)
- Support a new backbone: [DGCNN](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0.dev0/configs/dgcnn)
- Support 3D object detection on the S3DIS dataset
- Support compilation on Windows
- Full benchmark for PAConv on S3DIS
- Further enhancement for documentation, especially on the Chinese documentation

#### New Features

- Support 3D object detection on the S3DIS dataset (#835)
- Support PointRCNN (#842, #843, #856, #974, #1022, #1109, #1125)
- Support DGCNN (#896)
- Support PGD (#938, #940, #948, #950, #964, #1014, #1065, #1070, #1157)
- Support SMOKE (#939, #955, #959, #975, #988, #999, #1029)
- Support MonoFlex (#1026, #1044, #1114, #1115, #1183)
- Support CPU Training (#1196)

#### Improvements

- Support point sampling based on distance metric (#667, #840)
- Refactor coordinate systems (#677, #774, #803, #899, #906, #912, #968, #1001)
- Unify camera keys in PointFusion and transformations between different systems (#791, #805)
- Refine documentation (#792, #827, #829, #836, #849, #854, #859, #1111, #1113, #1116, #1121, #1132, #1135, #1185, #1193, #1226)
- Add a script to support benchmark regression (#808)
- Benchmark PAConvCUDA on S3DIS (#847)
- Support to download pdf and epub documentation (#850)
- Change the `repeat` setting in Group-Free-3D configs to reduce training epochs (#855)
- Support KITTI AP40 evaluation metric (#927)
- Add the mmdet3d2torchserve tool for SECOND (#977)
- Add code-spell pre-commit hook and fix typos (#995)
- Support the latest numba version (#1043)
- Set a default seed to use when the random seed is not specified (#1072)
- Distribute mix-precision models to each algorithm folder (#1074)
- Add abstract and a representative figure for each algorithm (#1086)
- Upgrade pre-commit hook (#1088, #1217)
- Support augmented data and ground truth visualization (#1092)
- Add local yaw property for `CameraInstance3DBoxes` (#1130)
- Lock the required numba version to 0.53.0 (#1159)
- Support the usage of plane information for KITTI dataset (#1162)
- Deprecate the support for "python setup.py test" (#1164)
- Reduce the number of multi-process threads to accelerate training (#1168)
- Support 3D flip augmentation for semantic segmentation (#1181)
- Update README format for each model (#1195)

#### Bug Fixes

- Fix compiling errors on Windows (#766)
- Fix the deprecated nms setting in the ImVoteNet config (#828)
- Use the latest `wrap_fp16_model` import from mmcv (#861)
- Remove 2D annotations generation on Lyft (#867)
- Update index files for the Chinese documentation to be consistent with the English version (#873)
- Fix the nested list transpose in the CenterPoint head (#879)
- Fix deprecated pretrained model loading for RegNet (#889)
- Fix the incorrect dimension indices of rotations and testing config in the CenterPoint test time augmentation (#892)
- Fix and improve visualization tools (#956, #1066, #1073)
- Fix PointPillars FLOPs calculation error (#1075)
- Fix missing dimension information in the SUN RGB-D data generation (#1120)
- Fix incorrect anchor range settings in the PointPillars [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/models/hv_pointpillars_secfpn_kitti.py) for KITTI (#1163)
- Fix incorrect model information in the RegNet metafile (#1184)
- Fix bugs in non-distributed multi-gpu training and testing (#1197)
- Fix a potential assertion error when generating corners from an empty box (#1212)
- Upgrade bazel version according to the requirement of Waymo Devkit (#1223)

#### Contributors

A total of 12 developers contributed to this release.

@THU17cyz, @wHao-Wu, @wangruohui, @Wuziyi616, @filaPro, @ZwwWayne, @Tai-Wang, @DCNSW, @xieenze, @robin-karlsson0, @ZCMax, @Otteri

### v0.18.1 (1/2/2022)

#### Improvements

- Support Flip3D augmentation in semantic segmentation task (#1182)
- Update regnet metafile (#1184)
- Add point cloud annotation tools introduction in FAQ (#1185)
- Add missing explanations of `cam_intrinsic` in the nuScenes dataset doc (#1193)

#### Bug Fixes

- Deprecate the support for "python setup.py test" (#1164)
- Fix the rotation matrix while rotation axis=0 (#1182)
- Fix the bug in non-distributed multi-gpu training/testing (#1197)
- Fix a potential bug when generating corners for empty bounding boxes (#1212)

#### Contributors

A total of 4 developers contributed to this release.

@ZwwWayne, @ZCMax, @Tai-Wang, @wHao-Wu

### v0.18.0 (1/1/2022)

#### Highlights

- Update the required minimum version of mmdet and mmseg

#### Improvements

- Use the official markdownlint hook and add codespell hook for pre-committing (#1088)
- Improve CI operation (#1095, #1102, #1103)
- Use shared menu content from OpenMMLab's theme and remove duplicated contents from config (#1111)
- Refactor the structure of documentation (#1113, #1121)
- Update the required minimum version of mmdet and mmseg (#1147)

#### Bug Fixes

- Fix symlink failure on Windows (#1096)
- Fix the upper bound of mmcv version in the mminstall requirements (#1104)
- Fix API documentation compilation and mmcv build errors (#1116)
- Fix figure links and pdf documentation compilation (#1132, #1135)

#### Contributors

A total of 4 developers contributed to this release.

@ZwwWayne, @ZCMax, @Tai-Wang, @wHao-Wu

### v0.17.3 (1/12/2021)

#### Improvements

- Change the default show value to `False` in show_result function to avoid unnecessary errors (#1034)
- Improve the visualization of detection results with colorized points in [single_gpu_test](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/apis/test.py#L11) (#1050)
- Clean unnecessary custom_imports in entrypoints (#1068)

#### Bug Fixes

- Update mmcv version in the Dockerfile (#1036)
- Fix the memory-leak problem when loading checkpoints in [init_model](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/apis/inference.py#L36) (#1045)
- Fix incorrect velocity indexing when formatting boxes on nuScenes (#1049)
- Explicitly set cuda device ID in [init_model](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/apis/inference.py#L36) to avoid memory allocation on unexpected devices (#1056)
- Fix PointPillars FLOPs calculation error (#1076)

#### Contributors

A total of 5 developers contributed to this release.

@wHao-Wu, @Tai-Wang, @ZCMax, @MilkClouds, @aldakata

### v0.17.2 (1/11/2021)

#### Improvements

- Update Group-Free-3D and FCOS3D bibtex (#985)
- Update the solutions for incompatibility of pycocotools in the FAQ (#993)
- Add Chinese documentation for the KITTI (#1003) and Lyft (#1010) dataset tutorial
- Add the H3DNet checkpoint converter for incompatible keys (#1007)

#### Bug Fixes

- Update mmdetection and mmsegmentation version in the Dockerfile (#992)
- Fix links in the Chinese documentation (#1015)

#### Contributors

A total of 4 developers contributed to this release.

@Tai-Wang, @wHao-Wu, @ZwwWayne, @ZCMax

### v0.17.1 (1/10/2021)

#### Highlights

- Support a faster but non-deterministic version of hard voxelization
- Completion of dataset tutorials and the Chinese documentation
- Improved the aesthetics of the documentation format

#### Improvements

- Add Chinese documentation for training on customized datasets and designing customized models (#729, #820)
- Support a faster but non-deterministic version of hard voxelization (#904)
- Update paper titles and code details for metafiles (#917)
- Add a tutorial for KITTI dataset (#953)
- Use Pytorch sphinx theme to improve the format of documentation (#958)
- Use the docker to accelerate CI (#971)

#### Bug Fixes

- Fix the sphinx version used in the documentation (#902)
- Fix a dynamic scatter bug that discards the first voxel by mistake when all input points are valid (#915)
- Fix the inconsistent variable names used in the [unit test](https://github.com/open-mmlab/mmdetection3d/blob/master/tests/test_models/test_voxel_encoder/test_voxel_generator.py) for voxel generator (#919)
- Upgrade to use `build_prior_generator` to replace the legacy `build_anchor_generator` (#941)
- Fix a minor bug caused by a too small difference set in the FreeAnchor Head (#944)

#### Contributors

A total of 8 developers contributed to this release.

@DCNSW, @zhanggefan, @mickeyouyou, @ZCMax, @wHao-Wu, @tojimahammatov, @xiliu8006, @Tai-Wang

### v0.17.0 (1/9/2021)

#### Compatibility

- Unify the camera keys for consistent transformation between coordinate systems on different datasets. The modification change the key names to `lidar2img`, `depth2img`, `cam2img`, etc. for easier understanding. Customized codes using legacy keys may be influenced.
- The next release will begin to move files of CUDA ops to [MMCV](https://github.com/open-mmlab/mmcv). It will influence the way to import related functions. We will not break the compatibility but will raise a warning first and please prepare to migrate it.

#### Highlights

- Support 3D object detection on the S3DIS dataset
- Support compilation on Windows
- Full benchmark for PAConv on S3DIS
- Further enhancement for documentation, especially on the Chinese documentation

#### New Features

- Support 3D object detection on the S3DIS dataset (#835)

#### Improvements

- Support point sampling based on distance metric (#667, #840)
- Update PointFusion to support unified camera keys (#791)
- Add Chinese documentation for customized dataset (#792), data pipeline (#827), customized runtime (#829), 3D Detection on ScanNet (#836), nuScenes (#854) and Waymo (#859)
- Unify camera keys used in transformation between different systems (#805)
- Add a script to support benchmark regression (#808)
- Benchmark PAConvCUDA on S3DIS (#847)
- Add a tutorial for 3D detection on the Lyft dataset (#849)
- Support to download pdf and epub documentation (#850)
- Change the `repeat` setting in Group-Free-3D configs to reduce training epochs (#855)

#### Bug Fixes

- Fix compiling errors on Windows (#766)
- Fix the deprecated nms setting in the ImVoteNet config (#828)
- Use the latest `wrap_fp16_model` import from mmcv (#861)
- Remove 2D annotations generation on Lyft (#867)
- Update index files for the Chinese documentation to be consistent with the English version (#873)
- Fix the nested list transpose in the CenterPoint head (#879)
- Fix deprecated pretrained model loading for RegNet (#889)

#### Contributors

A total of 11 developers contributed to this release.

@THU17cyz, @wHao-Wu, @wangruohui, @Wuziyi616, @filaPro, @ZwwWayne, @Tai-Wang, @DCNSW, @xieenze, @robin-karlsson0, @ZCMax

### v0.16.0 (1/8/2021)

#### Compatibility

- Remove the rotation and dimension hack in the monocular 3D detection on nuScenes by applying corresponding transformation in the pre-processing and post-processing. The modification only influences nuScenes coco-style json files. Please re-run the data preparation scripts if necessary. See more details in the PR #744.
- Add a new pre-processing module for the ScanNet dataset in order to support multi-view detectors. Please run the updated scripts to extract the RGB data and its annotations. See more details in the PR #696.

#### Highlights

- Support to use [MIM](https://github.com/open-mmlab/mim) with pip installation
- Support PAConv [models and benchmarks](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/paconv) on S3DIS
- Enhance the documentation especially on dataset tutorials

#### New Features

- Support RGB images on ScanNet for multi-view detectors (#696)
- Support FLOPs and number of parameters calculation (#736)
- Support to use [MIM](https://github.com/open-mmlab/mim) with pip installation (#782)
- Support PAConv models and benchmarks on the S3DIS dataset (#783, #809)

#### Improvements

- Refactor Group-Free-3D to make it inherit BaseModule from MMCV (#704)
- Modify the initialization methods of FCOS3D to be consistent with the refactored approach (#705)
- Benchmark the Group-Free-3D [models](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/groupfree3d) on ScanNet (#710)
- Add Chinese documentation for Getting Started (#725), FAQ (#730), Model Zoo (#735), Demo (#745), Quick Run (#746), Data Preparation (#787) and Configs (#788)
- Add documentation for semantic segmentation on ScanNet and S3DIS (#743, #747, #806, #807)
- Add a parameter `max_keep_ckpts` to limit the maximum number of saved Group-Free-3D checkpoints (#765)
- Add documentation for 3D detection on SUN RGB-D and nuScenes (#770, #793)
- Remove mmpycocotools in the Dockerfile (#785)

#### Bug Fixes

- Fix versions of OpenMMLab dependencies (#708)
- Convert `rt_mat` to `torch.Tensor` in coordinate transformation for compatibility (#709)
- Fix the `bev_range` initialization in `ObjectRangeFilter` according to the `gt_bboxes_3d` type (#717)
- Fix Chinese documentation and incorrect doc format due to the incompatible Sphinx version (#718)
- Fix a potential bug when setting `interval == 1` in [analyze_logs.py](https://github.com/open-mmlab/mmdetection3d/blob/master/tools/analysis_tools/analyze_logs.py) (#720)
- Update the structure of Chinese documentation (#722)
- Fix FCOS3D FPN BC-Breaking caused by the code refactoring in MMDetection (#739)
- Fix wrong `in_channels` when `with_distance=True` in the [Dynamic VFE Layers](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/models/voxel_encoders/voxel_encoder.py#L87) (#749)
- Fix the dimension and yaw hack of FCOS3D on nuScenes (#744, #794, #795, #818)
- Fix the missing default `bbox_mode` in the `show_multi_modality_result` (#825)

#### Contributors

A total of 12 developers contributed to this release.

@yinchimaoliang, @gopi231091, @filaPro, @ZwwWayne, @ZCMax, @hjin2902, @wHao-Wu, @Wuziyi616, @xiliu8006, @THU17cyz, @DCNSW, @Tai-Wang

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
- Support PyTorch 1.7 and 1.8
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
- Fix a compilation error triggered by [scatter_points_cuda.cu](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/src/scatter_points_cuda.cu) in PyTorch 1.7 (#393)

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
- Fix warning of deprecated usages of nonzero during training with PyTorch 1.6 (#330)

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
- Support docker with PyTorch 1.6.0
- Update baseline configs and results ([CenterPoint](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/centerpoint) on nuScenes and [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) on Waymo with full dataset)
- Switch model zoo to download.openmmlab.com

#### New Features

- Support dataset pipeline `VoxelBasedPointSampler` to sample multi-sweep points based on voxelization. (#125)
- Support mixed precision training of voxel-based methods (#132)
- Support docker with PyTorch 1.6.0 (#160)

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

- Fix all warnings and bugs in PyTorch 1.6.0 (#70, #72)
- Update issue templates (#43)
- Update unit tests (#20, #24, #30)
- Update documentation for using `ply` format point cloud data (#41)
- Use points loader to load point cloud data in ground truth (GT) samplers (#87)
- Unify version file of OpenMMLab projects by using `version.py` (#112)
- Remove unnecessary data preprocessing commands of SUN RGB-D dataset (#110)

#### Bug Fixes

- Rename CosineAnealing to CosineAnnealing (#57)
- Fix device inconsistent bug in 3D IoU computation (#69)
- Fix a minor bug in json2csv of lyft dataset (#78)
- Add missed test data for pointnet modules (#85)
- Fix `use_valid_flag` bug in `CustomDataset` (#106)

### v0.5.0 (9/7/2020)

MMDetection3D is released.
