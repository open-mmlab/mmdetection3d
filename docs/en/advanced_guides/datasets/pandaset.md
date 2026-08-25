# PandaSet Dataset

This page provides tutorials about the usage of MMDetection3D for PandaSet dataset.

## Overview

[PandaSet](https://pandaset.org/) is an autonomous driving dataset released by Hesai Technology and Scale AI. It features a **dual-LiDAR** system:

| Sensor   | Type                | Beams               | Coverage | Range |
| -------- | ------------------- | ------------------- | -------- | ----- |
| Pandar64 | Mechanical spinning | 64                  | 360°     | ~200m |
| PandarGT | Solid-state         | Dense forward array | ~60°     | ~200m |

The dataset contains 103 sequences × 80 frames = 8,240 frames, with 28 cuboid annotation categories.

## Before Preparation

Download PandaSet from [Kaggle](https://www.kaggle.com/datasets/usharengaraju/pandaset) or [ModelScope](https://www.modelscope.cn/) (~40 GB compressed, ~80 GB extracted).

Organize the folder structure as follows:

```
mmdetection3d
├── data
│   ├── pandaset
│   │   ├── sequences
│   │   │   ├── 001
│   │   │   │   ├── annotations
│   │   │   │   │   └── cuboids
│   │   │   │   │       ├── 00.pkl
│   │   │   │   │       ├── 01.pkl
│   │   │   │   │       └── ...
│   │   │   │   ├── lidar
│   │   │   │   │   ├── 00.pkl
│   │   │   │   │   ├── 01.pkl
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── poses.json
│   │   │   │   │   └── timestamps.json
│   │   │   │   └── meta
│   │   │   │       └── gps.json
│   │   │   ├── 002
│   │   │   └── ...
```

## Dataset Preparation

### Generate Info Files

Run the PandaSet converter to generate annotation pickle files:

```bash
python tools/dataset_converters/pandaset_converter.py \
    --pandaset-root data/pandaset/sequences \
    --out-dir data/pandaset \
    --sensors pandar64 pandargt \
    --workers 8
```

This produces:

```
data/pandaset/
├── pandaset_pandar64_infos_test.pkl   # 8240 frames, Pandar64 sensor
├── pandaset_pandargt_infos_test.pkl   # 8240 frames, PandarGT sensor
└── sequences/                          # Original data (symlink or actual)
```

### Coordinate Conventions

PandaSet stores annotations in **world coordinates**. The converter handles:

1. **World → Ego transform**: Using `lidar/poses.json` (4×4 matrix per frame)
2. **Yaw alignment**: PandaSet yaw=0 along X-axis → +π/2 offset for nuScenes model compatibility
3. **Z-center → Z-bottom**: PandaSet stores box center Z → subtract height/2 for MMDet3D convention

### Label Mapping

PandaSet has 28 cuboid categories. For cross-dataset evaluation with nuScenes models, they are mapped to 10 classes:

| nuScenes Class       | PandaSet Labels                                |
| -------------------- | ---------------------------------------------- |
| car                  | Car, Pickup Truck                              |
| truck                | Medium-sized Truck                             |
| trailer              | Towed Object                                   |
| bus                  | Bus / RV                                       |
| construction_vehicle | Construction Vehicle                           |
| bicycle              | Bicycle                                        |
| motorcycle           | Motorcycle                                     |
| pedestrian           | Pedestrian                                     |
| traffic_cone         | Cones                                          |
| barrier              | Road Barriers, Temporary Construction Barriers |

Unmapped PandaSet labels (e.g., "Animals", "Emergency Vehicle", "Train") are ignored.

## Evaluation

### Cross-Dataset Evaluation with nuScenes Model

PandaSet is primarily used for cross-dataset generalization testing. To evaluate a nuScenes-trained PointPillars model on PandaSet:

```bash
python tools/test.py \
    configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_pandaset-eval.py \
    checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth
```

For PandarGT (forward-facing only):

```bash
python tools/test.py \
    configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_pandaset-pandargt-eval.py \
    checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth
```

### Evaluation Metric

`PandaSetMetric` implements the nuScenes-style evaluation protocol:

- **Center-distance AP** at thresholds: 0.5m, 1.0m, 2.0m, 4.0m
- **True Positive errors**: mATE (translation), mASE (scale), mAOE (orientation)
- **NDS**: nuScenes Detection Score (partial — no velocity/attribute errors available)
- **Distance-bin analysis**: AP breakdown at 0–25m and 25–50m ranges
- **GT range filtering**: Only evaluates GT within the model's point_cloud_range

### Results

#### PointPillars FPN (nuScenes pretrained → PandaSet Pandar64)

| Metric         | Value | Notes                                   |
| -------------- | ----- | --------------------------------------- |
| mAP            | 0.072 | Cross-dataset (vs 0.40 on nuScenes val) |
| Car AP         | 0.47  | Best transferring class                 |
| Pedestrian AP  | 0.21  | Second best                             |
| NDS (partial)  | 0.18  | No velocity/attribute errors            |
| Car AP (0-25m) | 0.72  | Strong near-range performance           |

#### PointPillars FPN (nuScenes pretrained → PandaSet PandarGT)

| Metric | Value | Notes                                    |
| ------ | ----- | ---------------------------------------- |
| mAP    | 0.006 | PandarGT is incompatible with 360° model |
| Car AP | 0.05  | Severely degraded                        |

**Note:** PandarGT's 60° FOV is architecturally incompatible with models trained on 360° data. Only Pandar64 is suitable for cross-dataset evaluation with nuScenes/360° models.

### Why Cross-Dataset mAP is Low

The overall mAP (~7%) is significantly lower than nuScenes val (~40%) due to:

1. **Sensor difference**: Hesai Pandar64 (64-beam) vs Velodyne VLP-32C (32-beam) — different point density and beam patterns
2. **4th channel mismatch**: nuScenes model expects timestamp (≈0), PandaSet provides intensity (normalized to \[0,1\])
3. **Annotation mismatch**: Some classes have different physical definitions (e.g., "trailer" is 3.85m in PandaSet vs 12.3m in nuScenes)
4. **Size mismatch**: Objects like barriers have completely different dimensions between datasets

Classes that transfer well (car, pedestrian) have large, sensor-independent point cloud signatures. Classes that fail rely on sensor-specific point patterns or have annotation definition mismatches.

## Data Loading

PandaSet uses an on-the-fly loading approach via `LoadPointsFromPandaSet`:

1. Reads original `.pkl` files (Pandas DataFrames with columns: x, y, z, i, t, d)
2. Filters by sensor ID (d=0 for Pandar64, d=1 for PandarGT)
3. Transforms world coordinates to ego-centric frame using pose matrix
4. Normalizes intensity from \[0, 255\] to \[0, 1\]

This avoids pre-converting ~22 GB of point clouds to `.bin` format.

### Optional: Pre-conversion to .bin/.pcd

For workflows requiring standard file formats:

```bash
python tools/convert_pandaset_to_bin_pcd.py
```

This generates ego-centric `.bin` (float32) and `.pcd` files per frame per sensor, with corresponding `_bin` info pickle files for use with standard `LoadPointsFromFile`.

## Dual-LiDAR Notes

- **Pandar64** (360° spinning): Suitable for cross-dataset evaluation with any 360° LiDAR model
- **PandarGT** (60° forward): Only compatible with forward-facing or sector-based models; 360° models produce excessive false positives in empty BEV regions
