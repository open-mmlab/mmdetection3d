# FocalFormer3D: Focusing on Hard Instance for 3D Object Detection

> [FocalFormer3D: Focusing on Hard Instance for 3D Object Detection](https://arxiv.org/abs/2308.04556)

<!-- [ALGORITHM] -->

## Abstract

False negatives are a safety risk in autonomous driving, because a missed object is an object the planner never sees. FocalFormer3D attacks this directly with a multi-stage heatmap encoder that masks out whatever earlier stages already found, so each new stage is forced to look at the objects the previous ones missed. The queries accumulated this way go to a deformable transformer decoder with a rescoring head, which suppresses the extra false positives that harvesting hard instances would otherwise introduce. The method works for LiDAR-only and for LiDAR-camera fusion, and reaches 70.5 NDS on the nuScenes test set with LiDAR alone.

<div align=center>
<img src="https://github.com/NVlabs/FocalFormer3D/raw/master/docs/focalformer3d.png" width="800"/>
</div>

## Usage

### Compiling CUDA operations

The local attention and BEV pooling kernels build on first import through
`torch.utils.cpp_extension.load`, so no separate install step is needed. They do
require a visible GPU. Importing this plugin on a CPU-only machine raises
`AssertionError: local attention needs cuda!`, and the plugin turns that into a
`None` entry in the registry, which surfaces later as
`FocalFormer3D is not in the mmdet3d::model registry`. Run any registry check
inside a GPU job.

### Two version notes

This project needs mmcv 2.2.0 for the transformer layer API it builds on.
mmdet3d 1.4.0 sets `mmcv_maximum_version = '2.2.0'` and compares with a strict
`<`, so importing mmdet3d with mmcv 2.2.0 installed fails with
`MMCV==2.2.0 is used but incompatible`. Relax the bound in
`mmdet3d/__init__.py` locally before running anything here.

The released NVIDIA checkpoints were saved under the mmcv 0.x decoder layout
and their keys do not match mmdet 3.x. Convert once before use:

```bash
python projects/FocalFormer3D/tools/convert_focalformer_ckpt.py \
    --src FocalFormer3D_L_ep6_mAP664_NDS709.pth \
    --dst FocalFormer3D_L_converted.pth
```

### Training commands

```bash
bash tools/dist_train.sh projects/FocalFormer3D/configs/focalformer3d_lidar_voxel0075_second_secfpn_8xb2-cyclic-6e_nus-3d.py 8
```

The LiDAR-camera config is a fine-tuning stage. Upstream starts it from a
trained LiDAR checkpoint and a trained camera checkpoint, and freezes both
branches so only the fusion neck and the decoder learn. mmengine takes a single
`load_from`, so merge the two state dicts yourself and point `load_from` at the
result.

```bash
bash tools/dist_train.sh projects/FocalFormer3D/configs/focalformer3d_lidar-cam_voxel0075_second_secfpn_8xb2-cyclic-6e_nus-3d.py 8
```

### Testing commands

```bash
bash tools/dist_test.sh projects/FocalFormer3D/configs/focalformer3d_lidar_voxel0075_second_secfpn_8xb2-cyclic-6e_nus-3d.py ${CHECKPOINT_PATH} 8
```

## Box convention

The released checkpoints were trained under the mmdet3d 0.x nuScenes
convention, which writes box dimensions as (w, l, h) and the yaw as
`-yaw - pi/2` relative to what 1.x writes. The head learns whatever column
order it is trained on, so the detector converts at the boundary instead of
retraining the output layers. `loss` converts ground truth into the 0.x
convention on the way in and `add_pred_to_datasample` converts predictions back
on the way out. The yaw map is its own inverse, so both directions use the same
expression and the round trip is the identity. Training from scratch with these
configs is unaffected, since both conversions apply consistently.

## Results and models

### NuScenes

Evaluated on the nuScenes validation split with the released NVIDIA
checkpoints, converted as described above.

|                                               Modality                                               |     Config     | Voxel size | NDS  | mAP  |
| :--------------------------------------------------------------------------------------------------: | :------------: | :--------: | :--: | :--: |
|       [LiDAR](./configs/focalformer3d_lidar_voxel0075_second_secfpn_8xb2-cyclic-6e_nus-3d.py)        | FocalFormer3D  |   0.075    | 70.4 | 65.8 |
|      [LiDAR](./configs/deformformer3d_lidar_voxel0075_second_secfpn_8xb2-cyclic-20e_nus-3d.py)       | DeformFormer3D |   0.075    |  -   |  -   |
| [LiDAR + camera](./configs/focalformer3d_lidar-cam_voxel0075_second_secfpn_8xb2-cyclic-6e_nus-3d.py) | FocalFormer3D  |   0.075    |  -   |  -   |

The reference numbers for the LiDAR checkpoint are 70.9 NDS and 66.4 mAP. The
0.5 point gap comes from differences between the 0.x and 1.x data pipelines and
not from the model.

This port covers nuScenes only. Waymo support will follow in a separate PR if
this one is accepted.

## Citation

```latex
@inproceedings{chen2023focalformer3d,
  title={FocalFormer3D: Focusing on Hard Instance for 3D Object Detection},
  author={Chen, Yilun and Yu, Zhiding and Chen, Yukang and Lan, Shiyi and Anandkumar, Anima and Jia, Jiaya and Alvarez, Jose M},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

  - [x] Basic docstrings & proper citation

  - [x] Test-time correctness

  - [x] A full README

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

  - [ ] Unit tests

  - [ ] Code polishing

  - [ ] Metafile.yml

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
