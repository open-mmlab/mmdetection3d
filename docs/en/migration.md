Along with the release of OpenMMLab 2.0, MMDetection3D (namely MMDet3D) 1.x made many significant changes, resulting in less redundant, more efficient code and a more consistent overall design. These changes break backward compatibility. Therefore, we prepared this migration guide to make the transition as smooth as possible so that all users can enjoy the productivity benefits of the new MMDet3D and the entire OpenMMLab 2.0 ecosystem.

## Environment

MMDet3D 1.1 depends on the new foundational library for training deep learning models [MMEngine](https://github.com/open-mmlab/mmengine), and therefore has an entirely different dependency chain compared with MMDet3D 1.0. Even if you have a well-rounded MMDet3D 1.0 environment before, you still need to create a new python environment for MMDet3D 1.1. We provide a detailed [installation guide](./get_started.md) for reference.

## Dataset

You should update the annotation files generated in the old version since some key words and structures of annotation in MMDet3D 1.x have changed. The updating script is the following (taking KITTI as an xample):

```python
python tools/dataset_converters/update_infos_to_v2.py
        --dataset kitti
        --pkl-path ./data/kitti/kitti_infos_train.pkl
        --out-dir ./kitti_v2/
```

## Model

MMDet3D 1.x supports loading weights trained by the old version. For models that are important or highly used, we totally verified the precision of them in the 1.x version. Espectially for some models that has potential performance drop or training bugs in the old version, such as [centerpoint](https://github.com/open-mmlab/mmdetection3d/issues/2390), we check them and make sure the right precision in the new version. If you find any problem, please feel free to raise an [Issue](https://github.com/open-mmlab/mmdetection3d/issues). On the other hand, we also add some the latest SOTA motheds in our [package](../../configs/) and [projects](../../projects/). Thus, we strongly recommend you can use MMDet3D 1.1 to implement your project.
