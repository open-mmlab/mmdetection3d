# FireRPFNet Models - Quick Start Guide

Custom 3D object detection models using FireRPFNet architecture with Fire Modules, Residual connections, and CBAM attention.

## 🔥 FireRPFNet Variants

- **FireRPFNetV2**: Enhanced 3D LiDAR backbone with improved attention
- **FireRPFNet2D**: 2D image backbone variant for camera features

**Plug-and-Play Design:**
- **FireRPFNetV2** can replace SECOND backbone in any model (BEVFusion is one example shown here)
- **FireRPFNet2D** can be used as an efficient image backbone in multi-modal architectures
- Simply update the backbone config to integrate into your existing models

---

## 📋 Available Models

| Model | Config | Image Backbone | LiDAR Backbone | Dataset | Modality |
|-------|--------|---------------|----------------|---------|----------|
| MVXNet-Squeeze | `configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py` | SQUEEZE | **FireRPFNetV2** | KITTI | Multi-modal |
| MVXNet-Fire2D | `configs/mvxnet/mvxnet_firerpfnet2dfpn_fire_rpfnet_kitti-3d-3class.py` | **FireRPFNet2D** | **FireRPFNetV2** | KITTI | Multi-modal |
| BEVFusion-Lidar | `projects/BEVFusion/configs/bevfusion_lidar_voxel0075_firerpfnet_8xb4-cyclic-20e_nus-3d.py` | - | **FireRPFNetV2** | nuScenes | LiDAR-only |
| BEVFusion-Cam | `projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_firerpfnet_8xb4-cyclic-20e_nus-3d.py` | Swin-T | **FireRPFNetV2** | nuScenes | Multi-modal |

---

## 🚀 Installation

Follow the official MMDetection3D installation guide: https://mmdetection3d.readthedocs.io/en/latest/get_started.html

**Quick Setup:**
```bash
# Install dependencies
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'

# Install mmdetection3d
cd mmdetection3d
pip install -v -e .
```

---

## 📦 Dataset Setup

### KITTI (MVXNet models)
```bash
# Download from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
# Organize: data/kitti/training/{image_2, velodyne, calib, label_2}

# Create data infos
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

### nuScenes (BEVFusion models)
```bash
# Download from https://www.nuscenes.org/download
# Organize: data/nuscenes/{samples, sweeps, v1.0-trainval}

# Create data infos
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

---

## 🏋️ Training Commands

### MVXNet Models (KITTI)

**Model 1: SqueezeFPN + FireRPFNetV2**
```bash
# Single GPU
python tools/train.py configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py

```
- Batch size: 2/GPU | Epochs: 20 | LR: 0.001 | Val: Every 5 epochs

**Model 2: FireRPFNet2D + FireRPFNetV2**
```bash
# Single GPU
python tools/train.py configs/mvxnet/mvxnet_firerpfnet2dfpn_fire_rpfnet_kitti-3d-3class.py

```
- Batch size: 4/GPU | Epochs: 16 | LR: 0.001 | Val: Every 2 epochs | Early stopping enabled

---

### BEVFusion Models (nuScenes)

**Model 3: BEVFusion LiDAR-only + FireRPFNetV2**
```bash
# 
python tools/train.py projects/BEVFusion/configs/bevfusion_lidar_voxel0075_firerpfnet_8xb4-cyclic-20e_nus-3d.py
```
- Batch size: 4/GPU | Epochs: 20 | LR: 0.0002 | Cyclic scheduler

**Model 4: BEVFusion Multi-Modal + FireRPFNetV2**
```bash

# With mixed precision
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_firerpfnet_8xb4-cyclic-20e_nus-3d.py \
    --amp
```
- Batch size: 4/GPU (32 total) | Epochs: 6 | LR: 0.0002 | Val: Every epoch

---

## 🧪 Testing

### MVXNet Models
```bash
# Single GPU
python tools/test.py CONFIG CHECKPOINT

# Multi-GPU
bash tools/dist_test.sh CONFIG CHECKPOINT 4
```

**Examples:**
```bash
# Model 1
python tools/test.py \
    configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py \
    work_dirs/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class/best_checkpoint.pth

# Model 2
bash tools/dist_test.sh \
    configs/mvxnet/mvxnet_firerpfnet2dfpn_fire_rpfnet_kitti-3d-3class.py \
    work_dirs/mvxnet_firerpfnet2dfpn_fire_rpfnet_kitti-3d-3class/best_checkpoint.pth 4
```

### BEVFusion Models
```bash
# Model 3
bash tools/dist_test.sh \
    projects/BEVFusion/configs/bevfusion_lidar_voxel0075_firerpfnet_8xb4-cyclic-20e_nus-3d.py \
    work_dirs/bevfusion_lidar_voxel0075_firerpfnet_8xb4-cyclic-20e_nus-3d/best_checkpoint.pth 8

# Model 4
bash tools/dist_test.sh \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_firerpfnet_8xb4-cyclic-20e_nus-3d.py \
    work_dirs/bevfusion_lidar-cam_voxel0075_firerpfnet_8xb4-cyclic-20e_nus-3d/best_checkpoint.pth 8
```

---

## 💡 Tips

**Resume Training:**
```bash
python tools/train.py CONFIG --resume work_dirs/MODEL_NAME/epoch_X.pth
```

**Specify GPUs:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh CONFIG 4
```

**Debug Mode:**
```bash
python tools/train.py CONFIG \
    --cfg-options data.train_dataloader.num_workers=0 \
                  data.train_dataloader.batch_size=1
```

**Monitor Training:**
```bash
tensorboard --logdir=work_dirs/
```

---

## 🐛 Common Issues

**CUDA OOM:** Reduce batch size in config or via `--cfg-options data.train_dataloader.batch_size=1`

**Dataset not found:** Verify paths and run `python tools/create_data.py`

**Import errors:** Reinstall with `pip install -v -e .`

---

## 📚 References

- [MMDetection3D Documentation](https://mmdetection3d.readthedocs.io)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [nuScenes Dataset](https://www.nuscenes.org/)

---

## 📝 Citation

```bibtex
@article{firerpfnet2024,
  title={FireRPFNet: Efficient 3D Object Detection with Fire Modules and Attention},
  author={Aravind Singh},
  journal={arXiv preprint},
  year={2024}
}
```

---

**Happy Training! 🚀**

