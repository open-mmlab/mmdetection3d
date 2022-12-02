# Dummy ResNet Wrapper

This is an example README for community `projects/`. We have provided detailed explanations for each field in the form of html comments, which are visible when you read the source of this README file. If you wish to submit your project to our main repository, then all the fields in this README are mandatory for others to understand what you have achieved in this implementation.

## Description

<!-- Share any information you would like others to know. For example:
 Author: @xxx.
 This is an implementation of \[XXX\]. -->

This project implements a dummy ResNet wrapper, which literally does nothing new but prints "hello world" during initialization.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMDet3D's root directory, run the following command to train the model:

```bash
python tools/train.py projects/example_project/configs/fcos3d_dummy-resnet-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py
```

### Testing commands

In MMDet3D's root directory, run the following command to test the model:

```bash
python tools/test.py projects/example_project/configs/fcos3d_dummy-resnet-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py ${CHECKPOINT_PATH}
```

## Results

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmdetection3d/edit/dev-1.x/configs/fcos3d/README.md)
 You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

|                                                     Backbone                                                     | Lr schd | Mem (GB) | Inf time (fps) | mAP  | NDS  |         Download         |
| :--------------------------------------------------------------------------------------------------------------: | :-----: | :------: | :------------: | :--: | :--: | :----------------------: |
| [FCOS3D_dummy](projects/example_project/configs/fcos3d_dummy-resnet-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py) |   1x    |   8.69   |                | 29.8 | 37.7 | [model](<>) \| [log](<>) |

## Citation

<!-- You may remove this section if not applicable. -->

```latex
@inproceedings{wang2021fcos3d,
	title={{FCOS3D: Fully} Convolutional One-Stage Monocular 3D Object Detection},
	author={Wang, Tai and Zhu, Xinge and Pang, Jiangmiao and Lin, Dahua},
	booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
	year={2021}
}
# For the original 2D version
@inproceedings{tian2019fcos,
  title     =  {{FCOS: Fully} Convolutional One-Stage Object Detection},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      =  {2019}
}
```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.
 OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.
 Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
 A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmdet3d.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [ ] A full README

    <!-- As this template does. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/models/detectors/fcos_mono3d.py) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tests/test_models/test_dense_heads/test_fcos_mono3d_head.py) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/fcos3d/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](/configs/textdet/dbnet/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
