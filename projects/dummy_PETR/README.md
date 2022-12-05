# Dummy PETR

This is an README for `dummy_PETR`.

## Description

Author: @SekiroRong.
This is an implementation of *PETR*.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMDet3D's root directory, run the following command to train the model:

```bash
python tools/train.py projects/example_project/configs/projects/dummy_PETR/config/petr/petr_vovnet_gridmask_p4_800x320.py
```

### Testing commands

In MMDet3D's root directory, run the following command to test the model:

```bash
python tools/test.py projects/dummy_PETR/config/petr/petr_vovnet_gridmask_p4_800x320.py ${CHECKPOINT_PATH}
```

## Results

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmdetection3d/edit/dev-1.x/configs/fcos3d/README.md)
 You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

This Result is trained by petr_vovnet_gridmask_p4_800x320.py and use [weights](https://drive.google.com/file/d/1ABI5BoQCkCkP4B0pO5KBJ3Ni0tei0gZi/view?usp=sharing) as pretrain weight.

|                                                Backbone                                                | Lr schd | Mem (GB) | Inf time (fps) | mAP  | NDS  |         Download         |
| :----------------------------------------------------------------------------------------------------: | :-----: | :------: | :------------: | :--: | :--: | :----------------------: |
| [petr_vovnet_gridmask_p4_800x320](projects/dummy_PETR/configs/petr/petr_vovnet_gridmask_p4_800x320.py) |   1x    |   7.62   |      18.7      | 38.3 | 43.5 | [model](<>) \| [log](<>) |

```
mAP: 0.3827
mATE: 0.7375
mASE: 0.2703
mAOE: 0.4799
mAVE: 0.8699
mAAE: 0.2038
NDS: 0.4352
Eval time: 124.8s

Per-class results:
Object Class	  AP	  ATE	  ASE	  AOE	  AVE	  AAE
car	  0.574	  0.519	  0.150	  0.087	  0.865	  0.206
truck	  0.349	  0.773	  0.213	  0.117	  0.855	  0.220
bus	  0.423	  0.781	  0.204	  0.122	  1.902	  0.319
trailer 0.219	  1.034	  0.231	  0.608	  0.830	  0.149
construction_vehicle	  0.084	  1.062	  0.486	  1.245	  0.172	  0.360
pedestrian	  0.452	  0.681	  0.293	  0.646	  0.529	  0.231
motorcycle	  0.378	  0.670	  0.250	  0.567	  1.334	  0.130
bicycle	      0.347	  0.639	  0.264	  0.788	  0.472	  0.016
traffic_cone	  0.538	  0.553	  0.325	  nan	  nan	  nan
barrier	      0.464	  0.662	 0.287	  0.137	  nan	  nan
```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.
 OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.
 Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
 A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmdet3d.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [x] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [ ] A full README

    <!-- As this template does. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

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
