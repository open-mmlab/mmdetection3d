# MVXNet with custom backbones(SqueezeFPN + FireRPFNet): Efficient Multi-Modal 3D Object Detection

## Abstract

This project focuses on developing a computationally efficient and lightweight 3D object detection model for autonomous vehicles. By experimenting with and modifying the backbone architecture of existing fusion-based models, this project aims to reduce computational demand while maintaining or improving detection accuracy. The key research involved exploring the effectiveness of lightweight architectures like SqueezeNet, EfficientNet, and MobileNet for images, whereas VoxelNext and custom models, such as Residual Pilar Feature Network (RPFNet) and Fire Residual Pilar Feature Network (FireRPFNet) for LiDAR data processing. RPFNet and Fire RPFNet are our proposed solutions, which are inspired by fire module from SqueezeNet and Convolution Block Attention Module (CBAM), to processor lidar point cloud. The potential impact of this project extends to safer and more efficient autonomous driving, contributing to the advancement of intelligent transportation systems.

## Overview

Our model introduces a novel, computationally efficient approach to 3D object detection through several key innovations:

### Lightweight Image Processing
- **SqueezeFPN**: A highly efficient feature pyramid network adapted from SqueezeNet's architecture
- **Optimized Feature Extraction**: Carefully balanced network depth and width for optimal performance
- **Memory-Efficient Design**: Reduced parameter count while maintaining high feature quality

### Advanced LiDAR Processing
- **FireRPFNet**: Our custom-designed backbone network that combines:
  - Fire modules from SqueezeNet for efficient feature extraction
  - Convolution Block Attention Module (CBAM) for enhanced feature focus
  - Residual connections for improved gradient flow
- **Dynamic Voxel Feature Encoder**: Adaptive point cloud feature learning that adjusts to input density

### Efficient Multi-Modal Fusion
- Early fusion strategy for optimal feature integration
- Balanced computational resource allocation between modalities
- Adaptive feature aggregation for robust object detection

## Architecture

<div align=center>
The model consists of three main components:

1. **Image Branch**: 
   - SqueezeFPN backbone
   - Efficient feature pyramid with squeeze and expand layers
   - Multi-scale feature maps [512, 512, 512, 512]

2. **LiDAR Branch**:
   - Dynamic voxel feature encoder
   - FireRPFNet backbone with CBAM attention
   - Efficient feature processing with Fire modules

3. **Multi-modal Fusion**:
   - Early fusion of image and point cloud features
   - Point-wise feature fusion
   - Adaptive feature aggregation
</div>

## Results on KITTI Dataset

### AP@40 IoU Results (Car)

| Metric | Easy | Moderate | Hard |
|:------:|:----:|:--------:|:----:|
| 3D Detection | 97.37 | 91.93 | 89.53 |
| Bird's Eye View | 97.48 | 92.29 | 89.92 |
| 2D Detection | 95.39 | 88.64 | 84.56 |
| AOS | 94.92 | 87.22 | 82.64 |

### Comparison with State-of-the-Art

#### 3D Detection AP@40 (%)

| Model | Easy | Moderate | Hard |
|:-----:|:----:|:--------:|:----:|
| **Ours (SqueezeFPN+FireRPFNet)** | **97.37** | **91.93** | **89.53** |
| [TRTConv](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=30bdb9fd69e93886221650a590744f76bbb3d773) | 91.90 | 85.04 | 80.38 |
| [GLENet-VR](https://paperswithcode.com/paper/glenet-boosting-3d-object-detectors-with/review/?hl=60125) | 91.67 | 83.23 | 78.43 |
| [SE-SSD](https://paperswithcode.com/paper/cia-ssd-confident-iou-aware-single-stage/review/?hl=34051) | 91.49 | 82.54 | 77.15 |

#### Bird's Eye View AP@40 (%)

| Model | Easy | Moderate | Hard |
|:-----:|:----:|:--------:|:----:|
| **Ours (SqueezeFPN+FireRPFNet)** | **97.48** | **92.29** | **89.92** | [Pre-trained Model (25 epochs)](https://drive.google.com/file/d/19h9m7tb1bX-W1ZViocx4J8knanz6PuLz/view?usp=drive_link) |
| [SE-SSD](https://paperswithcode.com/paper/cia-ssd-confident-iou-aware-single-stage/review/?hl=34051) | 96.59 | 92.28 | 89.72 |
| [TRTConv](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=30bdb9fd69e93886221650a590744f76bbb3d773) | 95.55 | 92.04 | 87.23 |
| [PV-RCNN](https://paperswithcode.com/paper/pv-rcnn-point-voxel-feature-set-abstraction/review/?hl=13979) | 94.98 | 90.65 | 86.14 |

## Key Features

1. **Efficient Architecture**
   - Lightweight SqueezeFPN for image feature extraction
   - Memory-efficient FireRPFNet for point cloud processing
   - Dynamic voxel encoding for adaptive feature learning

2. **State-of-the-Art Performance**
   - Achieves top performance in both 3D detection and Bird's Eye View
   - Significant improvements over existing methods
   - Robust performance across different difficulty levels

3. **Multi-modal Fusion**
   - Early fusion strategy for better feature integration
   - Effective combination of image and LiDAR information
   - Enhanced feature representation for accurate detection

## Training and Testing

### Training
```shell
# Single GPU training
python tools/train.py configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py

# Multi-GPU training
TORCH_DISTRIBUTED_DEBUG=INFO ./tools/dist_train.sh configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py 8
```

### Testing
```shell
# Single GPU testing
python tools/test.py configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py ${CHECKPOINT_FILE}

# Multi-GPU testing
./tools/dist_test.sh configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py ${CHECKPOINT_FILE} 8
```

### inference command example 
update sample details before running command
```
python demo/multi_modality_demo.py \
    data/kitti/testing/velodyne/000068.bin \
    data/kitti/testing/image_2/000068.png \
    data/kitti/kitti_infos_test.pkl \
    work_dirs/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py \
    work_dirs/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class/epoch_25.pth
```

### Pre-trained Model

We provide a pre-trained model for 25 epochs on the KITTI dataset:
- [Download Pre-trained Model (25 epochs)](https://drive.google.com/file/d/19h9m7tb1bX-W1ZViocx4J8knanz6PuLz/view?usp=drive_link)



#### Usage Instructions
```shell
# Directly test the pre-trained model
python tools/test.py configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py /path/to/downloaded/checkpoint.pth

# Resume training from the pre-trained model
python tools/train.py configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py --resume /path/to/downloaded/checkpoint.pth
```

## Model Configuration

The model configuration can be found in:
```
configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py
```

Key configuration details:
- Backbone: SqueezeFPN + FireRPFNet
- Training schedule: 25 epochs with cosine learning rate
- Input: Multi-modal (LiDAR point cloud + Camera image)
- Classes: Pedestrian, Cyclist , Car
- Voxel size: [0.05, 0.05, 0.1]
- Point cloud range: [0, -40, -3, 70.4, 40, 1]

## Additional Resources

### Repository and Model Files
- [Complete Project Resources](https://drive.google.com/drive/folders/161GQlhavUursme3voNuQaF1YGtYi9E6a)
  - Pre-trained models
  - Logs
  - Additional configuration files
  - Supplementary materials

### Quick Access
- [25-Epoch Pre-trained Model](https://drive.google.com/file/d/19h9m7tb1bX-W1ZViocx4J8knanz6PuLz/view?usp=drive_link)
- Configuration File: `configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py`

**Note**: All resources are subject to the project's licensing terms. Please review and comply with usage guidelines.
