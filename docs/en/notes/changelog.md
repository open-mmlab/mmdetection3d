# Changelog of v1.1.x

### v1.1.0rc0 (1/9/2022)

We are excited to announce the release of MMDetection3D 1.1.0rc0.
MMDet3D 1.1.0rc0 is the first version of MMDetection3D 1.1.x, a part of the OpenMMLab 2.x projects.
Built upon the new [training engine](https://github.com/open-mmlab/mmengine),
MMDet3D 1.1.x unifies the interfaces of dataset, models, evaluation, and visualization with faster training and testing speed.
It also provides a standard protocol for different datasets, modalities and tasks.
We will support more strong baselines in the future release, with our latest exploration on camera-only 3D detection from videos.

### Highlights

1. **New engines**. MMDet3D 1.1.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a general and powerful runner that allows more flexible customizations and significantly simplifies the entrypoints of high-level interfaces.

2. **Unified interfaces**. As a part of the OpenMMLab 2.x projects, MMDet3D 1.1.x unifies and refactors the interfaces and internal logics of train, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.x projects share the same design in those interfaces and logics to allow the emergence of multi-task/modality algorithms.

3. **Standard protocol for all the datasets and modalities**. In addition to unified base datasets inherited from MMEngine, we also define the common keys across different datasets and unify all the info files with a standard protocol. It significantly simplifies the usage of multiple datasets and data modalities and paves the way for dataset customization. Please refer to the documentation of customized datasets for details.

4. **Strong baselines**. We will release strong baselines of many popular models to enable fair comparisons among state-of-the-art models.

5. **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmdetection3d.readthedocs.io/en/dev-1.x/).

### Breaking Changes

We briefly list the major breaking changes here. Please refer to the [compatibility documentation](./compatibility.md) and [migration guide](../migration.md) for details and migration instructions.

#### Training and testing

- MMDet3D 1.1.x runs on PyTorch>=1.6. We have deprecated the support of PyTorch 1.5 to embrace the mixed precision training and other new features since PyTorch 1.6. Some models can still run on PyTorch 1.5, but the full functionality of MMDet3D 1.1.x is not guaranteed.
- MMDet3D 1.1.x uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of dataset, model, evaluation, and visualizer. Therefore, MMDet3D 1.1.x no longer relies on the building logics of those modules in `mmdet.train.apis` and `tools/train.py`. Those code have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py).
- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic as that in training scripts to build the runner.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structures
- Config and model names

#### Components

- Datasets:
  - Refactor dataset classes to inherit from a unified `BaseDataset` in MMEngine
  - Define the common keys across different datasets and unify all the info files with a standard protocol
- Data Transforms: Refactor data transforms to inherit from basic transforms implemented in MMCV
- Models: Adjust the model interfaces to make them compatible with the latest data elements
- Evaluation: Decouple evaluators from datasets to make them more flexible
- Visualization: Design a unified visualizer based on MMEngine for different 3D tasks and settings

### New Features

1. Support a general semi-supervised learning framework that works with all the object detectors supported in MMDet 3.x. Please refer to [semi-supervised object detection](../user_guides/semi_det.md) for details.
2. Enable all the single-stage detectors to serve as region proposal networks. We give [an example of using FCOS as RPN](../user_guides/single_stage_as_rpn.md).
3. Support a semi-supervised object detection algorithm: [SoftTeacher](https://arxiv.org/abs/2106.09018).
4. Support [the updated CenterNet](https://arxiv.org/abs/2103.07461).
5. Support data structures `HorizontalBoxes` and `BaseBoxes` to encapsulate different kinds of bounding boxes. We are migrating to use data structures of boxes to replace the use of pure tensor boxes. This will unify the usages of different kinds of bounding boxes in MMDet 3.x and MMRotate 1.x to simplify the implementation and reduce redundant codes.

### Ongoing changes

1. Test-time augmentation: which is supported in MMDet3D 1.0.x, is not implemented in this version due to limited time slot. We will support it in the following releases with a new and simplified design.
2. Inference interfaces: a unified inference interfaces will be supported in the future to ease the use of released models.
3. Interfaces of useful tools that can be used in notebook: more useful tools that implemented in the `tools` directory will have their python interfaces so that they can be used through notebook and in downstream libraries.
4. Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMDet3D 1.1.x.
5. Support recent new features added in MMDet3D 1.0.x and our recent exploration on camera-only 3D detection from videos: we will help refactor these models and support them with benchmarks and models soon.

#### Contributors

A total of 6 developers contributed to this release.

@VVsssssk, @ZCMax, @ZwwWayne, @jshilong, @Tai-Wang, @lianqing11
