# Changelog of v1.1

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
