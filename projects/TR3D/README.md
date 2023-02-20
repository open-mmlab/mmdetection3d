# TR3D: Towards Real-Time Indoor 3D Object Detection

> [TR3D: Towards Real-Time Indoor 3D Object Detection](https://arxiv.org/abs/2302.02858)

## Abstract

Recently, sparse 3D convolutions have changed 3D object detection. Performing on par with the voting-based approaches, 3D CNNs are memory-efficient and scale to large scenes better. However, there is still room for improvement. With a conscious, practice-oriented approach to problem-solving, we analyze the performance of such methods and localize the weaknesses. Applying modifications that resolve the found issues one by one, we end up with TR3D: a fast fully-convolutional 3D object detection model trained end-to-end, that achieves state-of-the-art results on the standard benchmarks, ScanNet v2, SUN RGB-D, and S3DIS. Moreover, to take advantage of both point cloud and RGB inputs, we introduce an early fusion of 2D and 3D features. We employ our fusion module to make conventional 3D object detection methods multimodal and demonstrate an impressive boost in performance. Our model with early feature fusion, which we refer to as TR3D+FF, outperforms existing 3D object detection approaches on the SUN RGB-D dataset. Overall, besides being accurate, both TR3D and TR3D+FF models are lightweight, memory-efficient, and fast, thereby marking another milestone on the way toward real-time 3D object detection.

<div align="center">
<img src="https://user-images.githubusercontent.com/6030962/219644780-646516ec-a6c1-4ec5-9b8c-63bbc9702d05.png" width="800"/>
</div>

## Usage

Training and inference in this project were tested with `mmdet3d==1.1.0rc3`.

### Training commands

In MMDet3D's root directory, run the following command to train the model:

```bash
python tools/train.py projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py
```

### Testing commands

In MMDet3D's root directory, run the following command to test the model:

```bash
python tools/test.py projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py ${CHECKPOINT_PATH}
```

## Results and models

### ScanNet

|                          Backbone                          | Mem (GB) | Inf time (fps) |   AP@0.25   |   AP@0.5    |                                                                                                                                        Download                                                                                                                                         |
| :--------------------------------------------------------: | :------: | :------------: | :---------: | :---------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MinkResNet34](./configs/tr3d_1xb16_scannet-3d-18class.py) |   8.6    |      23.7      | 72.9 (72.0) | 59.3 (57.4) | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/tr3d/tr3d_1xb16_scannet-3d-18class/tr3d_1xb16_scannet-3d-18class.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/tr3d/tr3d_1xb16_scannet-3d-18class/tr3d_1xb16_scannet-3d-18class.log.json) |

### SUN RGB-D

|                          Backbone                          | Mem (GB) | Inf time (fps) |   AP@0.25   |   AP@0.5    |                                                                                                                                        Download                                                                                                                                         |
| :--------------------------------------------------------: | :------: | :------------: | :---------: | :---------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MinkResNet34](./configs/tr3d_1xb16_sunrgbd-3d-10class.py) |   3.8    |      27.5      | 67.1 (66.3) | 50.4 (49.6) | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/tr3d/tr3d_1xb16_sunrgbd-3d-10class/tr3d_1xb16_sunrgbd-3d-10class.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/tr3d/tr3d_1xb16_sunrgbd-3d-10class/tr3d_1xb16_sunrgbd-3d-10class.log.json) |

### S3DIS

|                        Backbone                         | Mem (GB) | Inf time (fps) |   AP@0.25   |   AP@0.5    |                                                                                                                                  Download                                                                                                                                   |
| :-----------------------------------------------------: | :------: | :------------: | :---------: | :---------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MinkResNet34](./configs/tr3d_1xb16_s3dis-3d-5class.py) |   15.2   |      21.0      | 74.5 (72.1) | 51.7 (47.6) | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/tr3d/tr3d_1xb16_s3dis-3d-5class/tr3d_1xb16_s3dis-3d-5class.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/tr3d/tr3d_1xb16_s3dis-3d-5class/tr3d_1xb16_s3dis-3d-5class.log.json) |

**Note**

- We report the results across 5 train runs followed by 5 test runs. Median values are in round brackets.
- Inference time is given for a single NVidia GeForce RTX 4090 GPU.

## Citation

```latex
@article{rukhovich2023tr3d,
  title={TR3D: Towards Real-Time Indoor 3D Object Detection},
  author={Rukhovich, Danila and Vorontsova, Anna and Konushin, Anton},
  journal={arXiv preprint arXiv:2302.02858},
  year={2023}
}
```

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

  - [x] Basic docstrings & proper citation

  - [x] Test-time correctness

  - [x] A full README

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

- [ ] Milestone 3: Good to be a part of our core package!

  - [x] Type hints and docstrings

  - [ ] Unit tests

  - [ ] Code polishing

  - [ ] Metafile.yml

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
