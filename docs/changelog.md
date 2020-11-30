## Changelog

### v0.8.0 (30/11/2020)

#### Highlights

- Refactor points structure with more constructive and clearer implementation.
- Support axis-aligned IoU loss for VoteNet with better performance.
- Update and enhance [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) benchmark on Waymo.

#### New Features

- Support axis-aligned IoU loss for VoteNet. (#194)
- Support points structure for consistent processing of all the point related representation. (#196, #204)

#### Improvements

- Enhance [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) benchmark on Waymo with stronger baselines. (#166)
- Add model zoo statistics and polish the documentation. (#201)

### v0.7.0 (1/11/2020)

#### Highlights

- Support a new method [SSN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700579.pdf) with benchmarks on nuScenes and Lyft datasets.
- Update benchmarks for SECOND on Waymo, CenterPoint with TTA on nuScenes and models with mixed precision training on KITTI and nuScenes.
- Support semantic segmentation on nuImages and provide [HTC](https://arxiv.org/abs/1901.07518) models with configurations and performance for reference.

#### Bug Fixes

- Fix incorrect code weights in anchor3d_head when introducing mixed precision training (#173)
- Fix the incorrect label mapping on nuImages dataset (#155)

#### New Features

- Modified primitive head which can support the setting on SUN-RGBD dataset (#136)
- Support semantic segmentation and [HTC](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/nuimages) with models for reference on nuImages dataset (#155)
- Support [SSN](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/ssn) on nuScenes and Lyft datasets (#147, #174, #166, #182)
- Support double flip for test time augmentation of CenterPoint with updated benchmark (#143)

#### Improvements

- Update [SECOND](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second) benchmark with configurations for reference on Waymo (#166)
- Delete checkpoints on Waymo to comply its specific license agreement (#180)
- Update models and instructions with [mixed precision training](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/fp16) on KITTI and nuScenes (#178)

### v0.6.1 (11/10/2020)

#### Highlights

- Support mixed precision training of voxel-based methods
- Support docker with pytorch 1.6.0
- Update baseline configs and results ([CenterPoint](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/centerpoint) on nuScenes and [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) on Waymo with full dataset)
- Switch model zoo to download.openmmlab.com

#### Bug Fixes

- Fix a bug of visualization in multi-batch case (#120)
- Fix bugs in dcn unit test (#130)
- Fix dcn bias bug in centerpoint (#137)
- Fix dataset mapping in the evaluation of nuScenes mini dataset (#140)
- Fix origin initialization in `CameraInstance3DBoxes` (#148, #150)
- Correct documentation link in the getting_started.md (#159)
- Fix model save path bug in gather_models.py (#153)
- Fix image padding shape bug in `PointFusion` (#162)

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

#### Bug Fixes

- Rename CosineAnealing to CosineAnnealing (#57)
- Fix device inconsistant bug in 3D IoU computation (#69)
- Fix a minor bug in json2csv of lyft dataset (#78)
- Add missed test data for pointnet modules (#85)
- Fix `use_valid_flag` bug in `CustomDataset` (#106)

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

### v0.5.0 (9/7/2020)

MMDetection3D is released.
