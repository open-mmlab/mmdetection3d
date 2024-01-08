# Changelog of v1.1

### v1.4.0 (8/1/2024)

#### Highlights

- Refactor Waymo dataset (#2836)
- Support the training of [DSVT](<(https://arxiv.org/abs/2301.06051)>) in `projects` (#2738)
- Support [Nerf-Det](https://arxiv.org/abs/2307.14620) in `projects` (#2732)

#### New Features

- Support the training of [DSVT](<(https://arxiv.org/abs/2301.06051)>) in `projects` (#2738)
- Support [Nerf-Det](https://arxiv.org/abs/2307.14620) in `projects` (#2732)
- Support [MV-FCOS3D++](https://arxiv.org/abs/2207.12716)
- Refactor Waymo dataset (#2836)

#### Improvements

- Support [PGD](https://arxiv.org/abs/2107.14160)) (front-of-view / multi-view) on Waymo dataset (#2835)
- Release new [Waymo-mini](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_mini.tar.gz) for verify some methods or debug quickly (#2835)

#### Bug Fixes

- Fix MinkUNet and SPVCNN some wrong configs (#2854)
- Fix incorrect number of arguments in PETR (#2800)
- Delete unused files in `mmdet3d/configs` (#2773)

#### Contributors

A total of 5 developers contributed to this release.

@sunjiahao1999, @WendellZ524, @Yanyirong, @JingweiZhang12, @Tai-Wang

### v1.3.0 (18/10/2023)

#### Highlights

- Support [CENet](https://arxiv.org/abs/2207.12691) in `projects` (#2619)
- Enhance demos with new 3D inferencers (#2763)

#### New Features

- Support [CENet](https://arxiv.org/abs/2207.12691) in `projects` (#2619)

#### Improvements

- Enhance demos with new 3D inferencers (#2763)
- Add BEV-based detection pipeline in nuScenes dataset tutorial (#2672)
- Add the new config type of Cylinder3D in `mmdet3d/configs` (#2681)
- Update [New Config Type](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta) (#2655)
- Update the QR code in README.md (#2703)

#### Bug Fixes

- Fix the download script of nuScenes dataset (#2660)
- Fix circleCI and GitHub workflow configuration (#2652)
- Fix the version of Open3D in requirements (#2633)
- Fix unused files in `mmdet3d/configs` (#2773)
- Fix support devices in FreeAnchor3DHead (#2769)
- Fix readthedocs building and link (#2739, #2650)
- Fix the pitch angle bug in LaserMix (#2710)

#### Contributors

A total of 6 developers contributed to this release.

@sunjiahao1999, @Xiangxu-0103, @ZhaoCake, @LRJKD, @crazysteeaam, @wep21, @zhiqwang

### v1.2.0 (4/7/2023)

#### Highlights

- Support [New Config Type](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta) in `mmdet3d/config`  (#2608)
- Support the inference of [DSVT](<(https://arxiv.org/abs/2301.06051)>) in `projects`  (#2606)
- Support downloading datasets from [OpenDataLab](https://opendatalab.com/) using `mim`  (#2593)

#### New Features

- Support [New Config Type](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta) in `mmdet3d/config`  (#2608)
- Support the inference of [DSVT](<(https://arxiv.org/abs/2301.06051)>) in `projects`  (#2606)
- Support downloading datasets from [OpenDataLab](https://opendatalab.com/) using `mim`  (#2593)

#### Improvements

- Enhanced visualization in interactive form (#2611)
- Update README.md and Model Zoo (#2599, #2600)
- Speed up S3DIS data preparation (#2585)

#### Bug Fixes

- Remove PointRCNN in benchmark training (#2610)
- Fix wrong indoor detection visualization (#2625)
- Fix MinkUNet download link (#2590)
- Fix the formula in the `readthedocs` (#2580)

#### Contributors

A total of 5 developers contributed to this release.

@sunjiahao1999, @Xiangxu-0103, @JingweiZhang12, @col14m, @zhulf0804

### v1.1.1 (30/5/2023)

#### Highlights

- Support [TPVFormer](https://arxiv.org/pdf/2302.07817.pdf) in `projects` (#2399, #2517, #2535)
- Support the training of BEVFusion in `projects` (#2546)
- Support lidar-based 3D semantic segmentation benchmark (#2530, #2559)

#### New Features

- Support [TPVFormer](https://arxiv.org/pdf/2302.07817.pdf) in `projects` (#2399, #2517, #2535)
- Support the training of \[BEVFusion\] in `projects` (#2558)
- Support lidar-based 3D Semantic Segmentation Benchmark (#2530, #2559)
- Support test-time augmentation for Segmentor (#2382)
- Support `Minkowski ConvModule` and `Residual` Block (#2528)
- Support the visualization of multi-view images in multi-modal methods (#2453)

#### Improvements

- Upload checkpoints and training log of PETR (#2555)
- Replace `np.float` by default `float` in segmentation evaluation (#2527)
- Add docs of converting SemanticKITTI datasets (#2515)
- Support different colors for different classes in visualization (#2500)
- Support tensor-like operations for `BaseInstance3DBoxes` and `BasePoint`
- Add information of LiDAR Segmentation in NuScenes annotation files
- Provide annotation files of datasets generated offline (#2457)
- Refactor document structure (#2429)
- Complete typehints and docstring (#2396, #2457, #2468, #2464, #2485)

#### Bug Fixes

- Fix the bug of abnormal loss when training SECOND in Automatic mixed precision(AMP) mode (#2452)
- Add a warning in function `post_process_coords` in mmdet3d/dataset/convert_utils.py (#2557)
- Fix invalid configs (#2477, #2536)
- Fix bugs of unit test (#2466)
- Update `local-rank` argument in test.py for pytorch 2.0 (#2469)
- Fix docker file (#2451)
- Fix demo and visualization (#2453)
- Fix SUN RGB-D data converter (#2440)
- Fix readthedocs building (#2459, #2419, #2505, #2396)
- Fix CI #(2445)
- Fix the version error of `torch` in github merge stage test (#2424)
- Loose the version restriction of `numba` (#2416)

#### Contributors

A total of 10 developers contributed to this release.

@sunjiahao1999, @Xiangxu-0103, @JingweiZhang12, @chriscarving, @jaan1729, @pd-michaelstanley, @filaPro, @kabouzeid, @A-new-b, @lbin

### v1.1.0 (6/4/2023)

#### Highlights

- Support [Cylinder3D](https://arxiv.org/pdf/2011.10033.pdf) (#2291, #2344, #2350)
- Support [MinkUnet](https://arxiv.org/abs/1904.08755) (#2294, #2358)
- Support [SPVCNN](https://arxiv.org/abs/2007.16100) (#2320，#2372)
- Support [TR3D](https://arxiv.org/abs/2302.02858) detector in `projects` (#2274)
- Support the inference of [BEVFusion](https://arxiv.org/abs/2205.13542) in `projects` (#2175)
- Support [DETR3D](https://arxiv.org/abs/2110.06922) in `projects` (#2173)

#### New Features

- Support [Cylinder3D](https://arxiv.org/pdf/2011.10033.pdf) (#2291, #2344, #2350)
- Support [MinkUnet](https://arxiv.org/abs/1904.08755) (#2294, #2358)
- Support [SPVCNN](https://arxiv.org/abs/2007.16100) (#2320，#2372)
- Support [TR3D](https://arxiv.org/abs/2302.02858) detector in `projects` (#2274)
- Support the inference of [BEVFusion](https://arxiv.org/abs/2205.13542) in `projects` (#2175)
- Support [DETR3D](https://arxiv.org/abs/2110.06922) in `projects` (#2173)
- Support PolarMix and LaserMix augmentation (#2265, #2302)
- Support loading annotation of panoptic segmentation (#2223)
- Support panoptic segmentation metric (#2230)
- Add inferencer for LiDAR-based, monocular and multi-modality 3D detection (#2208, #2190, #2342)
- Add inferencer for LiDAR-based segmentation (#2304)

#### Improvements

- Support `lazy_init` for CBGSDataset (#2271)
- Support generating annotation files for test set on Waymo  (#2180)
- Enhance the support for SemanticKitti (#2253, #2323)
- File I/O migration and reconstruction (#2319)
- Support `format_only` option for Lyft, NuScenes and Waymo datasets (#2333, #2151)
- Replace `np.transpose` with `torch.permute` to speed up (#2277)
- Allow setting local-rank for pytorch 2.0 (#2387)

#### Bug Fixes

- Fix the problem of reversal of length and width when drawing heatmap in CenterFormer (#2362)
- Deprecate old type alias due to the new version of numpy (#2339)
- Lose `trimesh` version requirements to fix numpy random state (#2340)
- Fix the device mismatch error in CenterPoint (#2308)
- Fix bug of visualization when there are no bboxes (#2231)
- Fix bug of counting ignore index in IOU in segmentation evaluation (#2229)

#### Contributors

A total of 14 developers contributed to this release.

@ZLTJohn, @SekiroRong, @shufanwu, @vansin, @triple-Mu, @404Vector, @filaPro, @sunjiahao1999, @Ginray, @Xiangxu-0103, @JingweiZhang12, @DezeZhao, @ZCMax, @roger-lcc

### v1.1.0rc3 (7/1/2023)

#### Highlights

- Support [CenterFormer](https://arxiv.org/abs/2209.05588) in `projects` (#2175)
- Support [PETR](https://arxiv.org/abs/2203.05625) in `projects` (#2173)

#### New Features

- Support [CenterFormer](https://arxiv.org/abs/2209.05588) in `projects` (#2175)
- Support [PETR](https://arxiv.org/abs/2203.05625) in `projects` (#2173)
- Refactor ImVoxelNet on SUN RGB-D into mmdet3d v1.1 (#2141)

#### Improvements

- Remove legacy builder.py (#2061)
- Update `customize_dataset` documentation (#2153)
- Update tutorial of LiDAR-based detection (#2120)

#### Bug Fixes

- Fix the configs of FCOS3D and PGD (#2191)
- Fix numpy's `ValueError` in update_infos_to_v2.py (#2162)
- Fix parameter missing in Det3DVisualizationHook (#2118)
- Fix memory overflow in the rotated box IoU calculation (#2134)
- Fix lidar2cam error in update_infos_to_v2.py for nus and lyft dataset (#2110)
- Fix error of data type in Waymo metrics (#2109)
- Update `bbox_3d` information in `cam_instances` for mono3d detection task (#2046)
- Fix label saving of Waymo dataset (#2096)

#### Contributors

A total of 10 developers contributed to this release.

@SekiroRong, @ZLTJohn, @vansin, @shanmo, @VVsssssk, @ZCMax, @Xiangxu-0103, @JingweiZhang12, @Tai-Wang, @lianqing11

### v1.1.0rc2 (2/12/2022)

#### Highlights

- Support [PV-RCNN](https://arxiv.org/abs/1912.13192)
- Speed up evaluation on Waymo dataset

#### New Features

- Support [PV-RCNN](https://arxiv.org/abs/1912.13192) (#1597, #2045)
- Speed up evaluation on Waymo dataset (#2008)
- Refactor FCAF3D into the framework of mmdet3d v1.1 (#1945)
- Refactor S3DIS dataset into the framework of mmdet3d v1.1 (#1984)
- Add `Projects/` folder and the first example project (#2042)

#### Improvements

- Rename `CLASSES` and `PALETTE` to `classes` and `palette` respectively (#1932)
- Update `metainfo` in pkl files and add `categories` into metainfo (#1934)
- Show instance statistics before and after through the pipeline (#1863)
- Add configs of DGCNN for different testing areas (#1967)
- Remove testing utils from `tests/utils/` to `mmdet3d/testing/` (#2012)
- Add typehint for code in `models/layers/` (#2014)
- Refine documentation (#1891, #1994)
- Refine voxelization for better speed (#2062)

#### Bug Fixes

- Fix loop visualization error about point cloud (#1914)
- Fix image conversion of Waymo to avoid information loss (#1979)
- Fix evaluation on KITTI testset (#2005)
- Fix sampling bug in `IoUNegPiecewiseSampler` (#2017)
- Fix point cloud range in CenterPoint (#1998)
- Fix some loading bugs and support FOV-image-based mode on Waymo dataset (#1942)
- Fix dataset conversion utils (#1923, #2040, #1971)
- Update metafiles in all the configs (#2006)

#### Contributors

A total of 12 developers contributed to this release.

@vavanade, @oyel, @thinkthinking, @PeterH0323， @274869388, @cxiang26, @lianqing11, @VVsssssk, @ZCMax, @Xiangxu-0103, @JingweiZhang12, @Tai-Wang

### v1.1.0rc1 (11/10/2022)

#### Highlights

- Support a camera-only 3D detection baseline on Waymo, [MV-FCOS3D++](https://arxiv.org/abs/2207.12716)

#### New Features

- Support a camera-only 3D detection baseline on Waymo, [MV-FCOS3D++](https://arxiv.org/abs/2207.12716), with new evaluation metrics and transformations (#1716)
- Refactor PointRCNN in the framework of mmdet3d v1.1 (#1819)

#### Improvements

- Add `auto_scale_lr` in config to support training with auto-scale learning rates (#1807)
- Fix CI (#1813, #1865, #1877)
- Update `browse_dataset.py` script (#1817)
- Update SUN RGB-D and Lyft datasets documentation (#1833)
- Rename `convert_to_datasample` to `add_pred_to_datasample` in detectors (#1843)
- Update customized dataset documentation (#1845)
- Update `Det3DLocalVisualization` and visualization documentation (#1857)
- Add the code of generating `cam_sync_labels` for Waymo dataset (#1870)
- Update dataset transforms typehints (#1875)

#### Bug Fixes

- Fix missing registration of models in [setup_env.py](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/utils/setup_env.py) (#1808)
- Fix the data base sampler bugs when using the ground plane data (#1812)
- Add output directory existing check during visualization (#1828)
- Fix bugs of nuScenes dataset for monocular 3D detection (#1837)
- Fix visualization hook to support the visualization of different data modalities (#1839)
- Fix monocular 3D detection demo (#1864)
- Fix the lack of `num_pts_feats` key in nuscenes dataset and complete docstring (#1882)

#### Contributors

A total of 10 developers contributed to this release.

@ZwwWayne, @Tai-Wang, @lianqing11, @VVsssssk, @ZCMax, @Xiangxu-0103, @JingweiZhang12, @tpoisonooo, @ice-tong, @jshilong

### v1.1.0rc0 (1/9/2022)

We are excited to announce the release of MMDetection3D 1.1.0rc0.
MMDet3D 1.1.0rc0 is the first version of MMDetection3D 1.1, a part of the OpenMMLab 2.0 projects.
Built upon the new [training engine](https://github.com/open-mmlab/mmengine) and [MMDet 3.x](https://github.com/open-mmlab/mmdetection/tree/3.x),
MMDet3D 1.1 unifies the interfaces of dataset, models, evaluation, and visualization with faster training and testing speed.
It also provides a standard data protocol for different datasets, modalities, and tasks for 3D perception.
We will support more strong baselines in the future release, with our latest exploration on camera-only 3D detection from videos.

### Highlights

1. **New engines**. MMDet3D 1.1 is based on [MMEngine](https://github.com/open-mmlab/mmengine) and [MMDet 3.x](https://github.com/open-mmlab/mmdetection/tree/3.x), which provides a universal and powerful runner that allows more flexible customizations and significantly simplifies the entry points of high-level interfaces.

2. **Unified interfaces**. As a part of the OpenMMLab 2.0 projects, MMDet3D 1.1 unifies and refactors the interfaces and internal logics of train, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.0 projects share the same design in those interfaces and logics to allow the emergence of multi-task/modality algorithms.

3. **Standard data protocol for all the datasets, modalities, and tasks for 3D perception**. Based on the unified base datasets inherited from MMEngine, we also design a standard data protocol that defines and unifies the common keys across different datasets, tasks, and modalities. It significantly simplifies the usage of multiple datasets and data modalities for multi-task frameworks and eases dataset customization. Please refer to the [documentation of customized datasets](../advanced_guides/customize_dataset.md) for details.

4. **Strong baselines**. We will release strong baselines of many popular models to enable fair comparisons among state-of-the-art models.

5. **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmdetection3d.readthedocs.io/en/1.1/).

### Breaking Changes

MMDet3D 1.1 has undergone significant changes to have better design, higher efficiency, more flexibility, and more unified interfaces.
Besides the changes of API, we briefly list the major breaking changes in this section.
We will update the [migration guide](../migration.md) to provide complete details and migration instructions.
Users can also refer to the [compatibility documentation](./compatibility.md) and [API doc](https://mmdetection3d.readthedocs.io/en/1.1/) for more details.

#### Dependencies

- MMDet3D 1.1 runs on PyTorch>=1.6. We have deprecated the support of PyTorch 1.5 to embrace the mixed precision training and other new features since PyTorch 1.6. Some models can still run on PyTorch 1.5, but the full functionality of MMDet3D 1.1 is not guaranteed.
- MMDet3D 1.1 relies on MMEngine to run. MMEngine is a new foundational library for training deep learning models of OpenMMLab and are widely depended by OpenMMLab 2.0 projects. The dependencies of file IO and training are migrated from MMCV 1.x to MMEngine.
- MMDet3D 1.1 relies on MMCV>=2.0.0rc0. Although MMCV no longer maintains the training functionalities since 2.0.0rc0, MMDet3D 1.1 relies on the data transforms, CUDA operators, and image processing interfaces in MMCV. Note that the package `mmcv` is the version that provides pre-built CUDA operators and `mmcv-lite` does not since MMCV 2.0.0rc0, while `mmcv-full` has been deprecated since 2.0.0rc0.
- MMDet3D 1.1 is based on MMDet 3.x, which is also a part of OpenMMLab 2.0 projects.

#### Training and testing

- MMDet3D 1.1 uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of dataset, model, evaluation, and visualizer. Therefore, MMDet3D 1.1 no longer relies on the building logics of those modules in `mmdet3d.train.apis` and `tools/train.py`. Those code have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py). Please refer to the [migration guide of Runner in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for more details.
- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic as that in training scripts to build the runner.
- The execution points of hooks in the new Runner have been enriched to allow more flexible customization. Please refer to the [migration guide of Hook in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/hook.html) for more details.
- Learning rate and momentum scheduling has been migrated from Hook to [Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html). Please refer to the [migration guide of Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html) for more details.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structure to ease the understanding of the components in runner. Users can read the [config example of MMDet3D 1.1](../user_guides/config.md) or refer to the [migration guide in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for migration details.
- The file names of configs and models are also refactored to follow the new rules unified across OpenMMLab 2.0 projects. The names of checkpoints are not updated for now as there is no BC-breaking of model weights between MMDet3D 1.1 and 1.0.x. We will progressively replace all the model weights by those trained in MMDet3D 1.1. Please refer to the [user guides of config](../user_guides/config.md) for more details.

#### Dataset

The Dataset classes implemented in MMDet3D 1.1 all inherits from the `Det3DDataset` and `Seg3DDataset`, which inherits from the [BaseDataset in MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html). In addition to the changes of interfaces, there are several changes of Dataset in MMDet3D 1.1.

- All the datasets support to serialize the internal data list to reduce the memory when multiple workers are built for data loading.
- The internal data structure in the dataset is changed to be self-contained (without losing information like class names in MMDet3D 1.0.x) while keeping simplicity.
- Common keys across different datasets and data modalities are defined and all the info files are unified into a standard protocol.
- The evaluation functionality of each dataset has been removed from dataset so that some specific evaluation metrics like KITTI AP can be used to evaluate the prediction on other datasets.

#### Data Transforms

The data transforms in MMDet3D 1.1 all inherits from `BaseTransform` in MMCV>=2.0.0rc0, which defines a new convention in OpenMMLab 2.0 projects.
Besides the interface changes, there are several changes listed as below:

- The functionality of some data transforms (e.g., `Resize`) are decomposed into several transforms to simplify and clarify the usages.
- The format of data dict processed by each data transform is changed according to the new data structure of dataset.
- Some inefficient data transforms (e.g., normalization and padding) are moved into data preprocessor of model to improve data loading and training speed.
- The same data transforms in different OpenMMLab 2.0 libraries have the same augmentation implementation and the logic given the same arguments, i.e., `Resize` in MMDet 3.x and MMSeg 1.x will resize the image in the exact same manner given the same arguments.

#### Model

The models in MMDet3D 1.1 all inherits from `BaseModel` in MMEngine, which defines a new convention of models in OpenMMLeb 2.0 projects.
Users can refer to [the tutorial of model in MMengine](https://mmengine.readthedocs.io/en/latest/tutorials/model.html) for more details.
Accordingly, there are several changes as the following:

- The model interfaces, including the input and output formats, are significantly simplified and unified following the new convention in MMDet3D 1.1.
  Specifically, all the input data in training and testing are packed into `inputs` and `data_samples`, where `inputs` contains model inputs like a dict contain a list of image tensors and the point cloud data, and `data_samples` contains other information of the current data sample such as ground truths, region proposals, and model predictions. In this way, different tasks in MMDet3D 1.1 can share the same input arguments, which makes the models more general and suitable for multi-task learning and some flexible training paradigms like semi-supervised learning.
- The model has a data preprocessor module, which are used to pre-process the input data of model. In MMDet3D 1.1, the data preprocessor usually does necessary steps to form the input images into a batch, such as padding. It can also serve as a place for some special data augmentations or more efficient data transformations like normalization.
- The internal logic of model have been changed. In MMDet3D 1.1, model uses `forward_train`, `forward_test`, `simple_test`, and `aug_test` to deal with different model forward logics. In MMDet3D 1.1 and OpenMMLab 2.0, the forward function has three modes: 'loss', 'predict', and 'tensor' for training, inference, and tracing or other purposes, respectively.
  The forward function calls `self.loss`, `self.predict`, and `self._forward` given the modes 'loss', 'predict', and 'tensor', respectively.

#### Evaluation

The evaluation in MMDet3D 1.0.x strictly binds with the dataset. In contrast, MMDet3D 1.1 decomposes the evaluation from dataset, so that all the detection dataset can evaluate with KITTI AP and other metrics implemented in MMDet3D 1.1.
MMDet3D 1.1 mainly implements corresponding metrics for each dataset, which are manipulated by [Evaluator](https://mmengine.readthedocs.io/en/latest/design/evaluator.html) to complete the evaluation.
Users can build evaluator in MMDet3D 1.1 to conduct offline evaluation, i.e., evaluate predictions that may not produced in MMDet3D 1.1 with the dataset as long as the dataset and the prediction follows the dataset conventions. More details can be find in the [tutorial in mmengine](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html).

#### Visualization

The functions of visualization in MMDet3D 1.1 are removed. Instead, in OpenMMLab 2.0 projects, we use [Visualizer](https://mmengine.readthedocs.io/en/latest/design/visualization.html) to visualize data. MMDet3D 1.1 implements `Det3DLocalVisualizer` to allow visualization of 2D and 3D data, ground truths, model predictions, and feature maps, etc., at any place. It also supports to send the visualization data to any external visualization backends such as Tensorboard.

### Planned changes

We list several planned changes of MMDet3D 1.1.0rc0 so that the community could more comprehensively know the progress of MMDet3D 1.1. Feel free to create a PR, issue, or discussion if you are interested, have any suggestions and feedbacks, or want to participate.

1. Test-time augmentation: which is supported in MMDet3D 1.0.x, is not implemented in this version due to limited time slot. We will support it in the following releases with a new and simplified design.
2. Inference interfaces: a unified inference interfaces will be supported in the future to ease the use of released models.
3. Interfaces of useful tools that can be used in notebook: more useful tools that implemented in the `tools` directory will have their python interfaces so that they can be used through notebook and in downstream libraries.
4. Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMDet3D 1.1.
5. Wandb visualization: MMDet 2.x supports data visualization since v2.25.0, which has not been migrated to MMDet 3.x for now. Since Wandb provides strong visualization and experiment management capabilities, a `DetWandbVisualizer` and maybe a hook are planned to fully migrated those functionalities in MMDet 2.x and a `Det3DWandbVisualizer` will be supported in MMDet3D 1.1 accordingly.
6. Will support recent new features added in MMDet3D 1.0.x and our recent exploration on camera-only 3D detection from videos: we will refactor these models and support them with benchmarks and models soon.
