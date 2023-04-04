Along with the release of OpenMMLab 2.0, MMDetection3D (namely MMDet3D) 1.1 made many significant changes, resulting in less redundant, more efficient code and a more consistent overall design. These changes break backward compatibility. Therefore, we prepared this migration guide to make the transition as smooth as possible so that all users can enjoy the productivity benefits of the new MMDet3D and the entire OpenMMLab 2.0 ecosystem.

## Environment

MMDet3D 1.1 depends on the new foundational library for training deep learning models [MMEngine](https://github.com/open-mmlab/mmengine), and therefore has an entirely different dependency chain compared with MMDet3D 1.0. Even if you have a well-rounded MMDet3D 1.0 / 0.x environment before, you still need to create a new python environment for MMDet3D 1.1. We provide a detailed [installation guide](./get_started.md) for reference.

## Dataset

You should update the annotation files generated in the 1.0 version since some key words and structures of annotation in MMDet3D 1.1 have changed. The updating script is the following (taking KITTI as an xample):

```python
python tools/dataset_converters/update_infos_to_v2.py
        --dataset kitti
        --pkl-path ./data/kitti/kitti_infos_train.pkl
        --out-dir ./kitti_v2/
```

If your annotation files are generated in the 0.x version, you should firstly update them to 1.0 version using this [script](../../tools/update_data_coords.py). Alternatively, you can re-genetate annotation files from scratch use this [script](../../tools/create_data.py).

## Model

MMDet3D 1.1 supports loading weights trained by the old version (1.0 version). For models that are important or frequently used, we have thoroughly verified the precision of them in the 1.1 version. Espectially for some models that may experience potential performance drop or training bugs in the old version, such as [centerpoint](https://github.com/open-mmlab/mmdetection3d/issues/2390), we have checked them and ensured the right precision in the new version. If you encounter any problem, please feel free to raise an [Issue](https://github.com/open-mmlab/mmdetection3d/issues). Additionally, we have added some of the latest SOTA motheds in our [package](../../configs/) and [projects](../../projects/), making MMDet3D 1.1 a highly recommended choice for implementing your project.
