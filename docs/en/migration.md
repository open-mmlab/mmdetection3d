Along with the release of OpenMMLab 2.0, MMDetection3D (namely MMDet3D) 1.1 made many significant changes, resulting in less redundant, more efficient code and a more consistent overall design. These changes break backward compatibility. Therefore, we prepared this migration guide to make the transition as smooth as possible so that all users can enjoy the productivity benefits of the new MMDet3D and the entire OpenMMLab 2.0 ecosystem.

## Environment

MMDet3D 1.1 depends on the new foundational library [MMEngine](https://github.com/open-mmlab/mmengine) for training deep learning models, and therefore has an entirely different dependency chain compared with MMDet3D 1.0. Even if you have a well-rounded MMDet3D 1.0 / 0.x environment before, you still need to create a new Python environment for MMDet3D 1.1. We provide a detailed [installation guide](./get_started.md) for reference.

The configuration files in our new version have a lot of modifications because of the differences between MMCV 1.x and MMEngine. The guides for migration from MMCV to MMEngine can be seen [here](https://github.com/open-mmlab/mmengine/tree/main/docs/en/migration).

We have renamed the names of the remote branches in MMDet3D 1.1 (renaming 1.1 to main, master to 1.0, and dev to dev-1.0). If your local branches in the git system are not aligned with branches of the remote repo, you can use the following commands to resolve it:

```
git fetch origin
git checkout main
git branch main_backup  # backup your main branch
git reset --hard origin/main
```

## Dataset

You should update the annotation files generated in the 1.0 version since some key words and structures of annotation in MMDet3D 1.1 have changed. Taking KITTI as an example, the update script is as follows:

```python
python tools/dataset_converters/update_infos_to_v2.py
        --dataset kitti
        --pkl-path ./data/kitti/kitti_infos_train.pkl
        --out-dir ./kitti_v2/
```

If your annotation files are generated in the 0.x version, you should first update them to 1.0 version using this [script](../../tools/update_data_coords.py). Alternatively, you can re-generate annotation files from scratch using this [script](../../tools/create_data.py).

## Model

MMDet3D 1.1 supports loading weights trained on the old version (1.0 version). For models that are important or frequently used, we have thoroughly verified their precisions in the 1.1 version. Especially for some models that may experience potential performance drop or training bugs in the old version, such as [centerpoint](https://github.com/open-mmlab/mmdetection3d/issues/2390), we have checked them and ensured the right precision in the new version. If you encounter any problem, please feel free to raise an [issue](https://github.com/open-mmlab/mmdetection3d/issues). Additionally, we have added some of the latest SOTA methods in our [package](../../configs/) and [projects](../../projects/), making MMDet3D 1.1 a highly recommended choice for implementing your project.
