# Dataset Preparation

## Before Preparation

It is recommended to symlink the dataset root to `$MMDETECTION3D/data`.
If your folder structure is different from the following, you may need to change the corresponding paths in config files.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
│   ├── waymo
│   │   ├── waymo_format
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
│   │   │   ├── gt.bin
│   │   ├── kitti_format
│   │   │   ├── ImageSets
│   ├── lyft
│   │   ├── v1.01-train
│   │   │   ├── v1.01-train (train_data)
│   │   │   ├── lidar (train_lidar)
│   │   │   ├── images (train_images)
│   │   │   ├── maps (train_maps)
│   │   ├── v1.01-test
│   │   │   ├── v1.01-test (test_data)
│   │   │   ├── lidar (test_lidar)
│   │   │   ├── images (test_images)
│   │   │   ├── maps (test_maps)
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   ├── sample_submission.csv
│   ├── s3dis
│   │   ├── meta_data
│   │   ├── Stanford3dDataset_v1.2_Aligned_Version
│   │   ├── collect_indoor3d_data.py
│   │   ├── indoor3d_util.py
│   │   ├── README.md
│   ├── scannet
│   │   ├── meta_data
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── batch_load_scannet_data.py
│   │   ├── load_scannet_data.py
│   │   ├── scannet_utils.py
│   │   ├── README.md
│   ├── sunrgbd
│   │   ├── OFFICIAL_SUNRGBD
│   │   ├── matlab
│   │   ├── sunrgbd_data.py
│   │   ├── sunrgbd_utils.py
│   │   ├── README.md
│   ├── semantickitti
│   │   ├── sequences
│   │   │   ├── 00
│   │   │   │   ├── labels
│   │   │   │   ├── velodyne
│   │   │   ├── 01
│   │   │   ├── ..
│   │   │   ├── 22

```

## Download and Data Preparation

### KITTI

1. Download KITTI 3D detection data [HERE](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Alternatively, you
   can download the dataset from [OpenDataLab](https://opendatalab.com/) using MIM. The command scripts are the following:

```bash
# install OpenDataLab CLI tools
pip install -U opendatalab
# log in OpenDataLab. Note that you should register an account on [OpenDataLab](https://opendatalab.com/) before.
pip install odl
odl login
# download and preprocess by MIM
mim download mmdet3d --dataset kitti
```

2. Prepare KITTI data splits by running:

```bash
mkdir ./data/kitti/ && mkdir ./data/kitti/ImageSets

# Download data split
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt
```

3. Generate info files by running:

```bash
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

In an environment using slurm, users may run the following command instead:

```bash
sh tools/create_data.sh <partition> kitti
```

**Tips**:

- **Ready-made Annotations**. We have also provided kitti data annotation files generated offline [here](#summary-of-annotation-files). You could download them and place them under `data/kitti/`. However, if you want to use `ObjectSample` Augmentation in LiDAR-based detection methods, you should additionally generate groundtruth database files and annotations.

  ```bash
  python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti --only-gt-database
  ```

### Waymo

Download Waymo open dataset V1.4.1 [HERE](https://waymo.com/open/download/) and its data split [HERE](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o?usp=sharing). Then put `.tfrecord` files into corresponding folders in `data/waymo/waymo_format/` and put the data split `.txt` files into `data/waymo/kitti_format/ImageSets`. Download ground truth `.bin` file for validation set [HERE](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0/validation/ground_truth_objects) and put it into `data/waymo/waymo_format/`. A tip is that you can use `gsutil` to download the large-scale dataset with commands. You can take this [tool](https://github.com/RalphMao/Waymo-Dataset-Tool) as an example for more details. Subsequently, prepare waymo data by running:

```bash
# TF_CPP_MIN_LOG_LEVEL=3 will disable all logging output from TensorFlow.
# The number of `--workers` depends on the maximum number of cores in your CPU.
TF_CPP_MIN_LOG_LEVEL=3 python tools/create_data.py waymo --root-path ./data/waymo --out-dir ./data/waymo --workers 128 --extra-tag waymo --version v1.4
```

Note that:

- In case the preprocessing of Waymo dataset is slow or blocked, consider reducing the value of `--workers`. If this doesn't resolve the issue, you could set `--workers` as 0 to avoid using multiprocess.

- If your local disk does not have enough space for saving converted data, you can change the `--out-dir` to anywhere else. Just remember to create folders and prepare data there in advance and link them back to `data/waymo/kitti_format` after the data conversion.

**Tips**:

- **Ready-made Annotations**. We have provided the annotation files generated offline [here](#summary-of-annotation-files). However, the original Waymo data still needs to be converted to `kitti-format` data by yourself.

- **Waymo-mini**. If you just want to use a part of Waymo Dataset to verify some methods or debug quickly, you could use our provided [Waymo-mini](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_mini.tar.gz) which only contains two segments in train split and one segment in val split from the original dataset. All the images, point clouds and annotations in this compressed file have been processed offline so that you can directly download and unzip it to `data/waymo/`:

  ```bash
  tar -xzvf waymo_mini.tar.gz -C ./data/waymo_mini
  ```

### NuScenes

1. Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). Alternatively, you
   can download the dataset from [OpenDataLab](https://opendatalab.com/) using MIM. The downloading and unzipping command scripts are the following:

```bash
# install OpenDataLab CLI tools
pip install -U opendatalab
# log in OpenDataLab. Note that you should register an account on [OpenDataLab](https://opendatalab.com/) before.
pip install odl
odl login
# download and preprocess by MIM
mim download mmdet3d --dataset nuscenes
```

2. Prepare nuscenes data by running:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

**Tips**:

- **Ready-made Annotations**. We have also provided NuScenes data annotation files generated offline [here](#summary-of-annotation-files). You could download them and place them under `data/nuscenes/`. However, if you want to use `ObjectSample` Augmentation in LiDAR-based detection methods, you should additionally generate groundtruth database files and annotations.

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --only-gt-database
```

### Lyft

Download Lyft 3D detection data [HERE](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data). Prepare Lyft data by running:

```bash
python tools/create_data.py lyft --root-path ./data/lyft --out-dir ./data/lyft --extra-tag lyft --version v1.01
python tools/dataset_converters/lyft_data_fixer.py --version v1.01 --root-folder ./data/lyft
```

Note that we follow the original folder names for clear organization. Please rename the raw folders as shown above. Also note that the second command serves the purpose of fixing a corrupted lidar data file. Please refer to the [discussion](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110000) for more details.

### SemanticKITTI

1. Download SemanticKITTI dataset [HERE](http://semantic-kitti.org/dataset.html#download) and unzip all zip files. Alternatively, you
   can download the dataset from [OpenDataLab](https://opendatalab.com/) using MIM. The downloading and unzipping command scripts are the following:

```bash
# install OpenDataLab CLI tools
pip install -U opendatalab
# log in OpenDataLab. Note that you should register an account on [OpenDataLab](https://opendatalab.com/) before.
pip install odl
odl login
# download and preprocess by MIM
mim download mmdet3d --dataset semantickitti
```

2. Generate info files by running:

```bash
python ./tools/create_data.py semantickitti --root-path ./data/semantickitti --out-dir ./data/semantickitti --extra-tag semantickitti
```

**Tips**:

- **Ready-made Annotations**. We have also provided SemanticKITTI data annotation files generated offline [here](#summary-of-annotation-files). You could download them and place them under `data/semantickitti/`.

### S3DIS, ScanNet and SUN RGB-D

To prepare S3DIS data, please see its [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/s3dis/README.md).

To prepare ScanNet data, please see its [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/scannet/README.md).

To prepare SUN RGB-D data, please see its [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/sunrgbd/README.md).

**Tips**: For S3DIS, ScanNet and SUN RGB-D datasets, we have also provided data annotation files generated offline [here](#summary-of-annotation-files). You could download them and place them under `data/${DATASET}/`. However, you also need to generate point cloud files and semantic/instance masks files (if it has) by yourself.

### Customized Datasets

For using custom datasets, please refer to [Customize Datasets](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docs/en/advanced_guides/customize_dataset.md).

### Update data infos

If you have used v1.0.0rc1-v1.0.0rc4 mmdetection3d to create data infos before, and now you want to use the newest v1.1.0 mmdetection3d, you need to update the data infos file.

```bash
python tools/dataset_converters/update_infos_to_v2.py --dataset ${DATA_SET} --pkl-path ${PKL_PATH} --out-dir ${OUT_DIR}
```

- `--dataset` : Name of dataset.
- `--pkl-path` : Specify the data infos pkl file path.
- `--out-dir` : Output direction of the data infos pkl file.

Example:

```bash
python tools/dataset_converters/update_infos_to_v2.py --dataset kitti --pkl-path ./data/kitti/kitti_infos_trainval.pkl --out-dir ./data/kitti
```

### Summary of annotation files

We provide ready-made annotation files we generated offline for reference. You can directly use these files for convenice.

|                                                  Dataset                                                  |                                                                                                           Train annotation file                                                                                                           |                                                                                                        Val annotation file                                                                                                         |                                                                                                              Test information file                                                                                                              |
| :-------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                   KITTI                                                   |                                                                  [kitti_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/kitti/kitti_infos_train.pkl)                                                                   |                                                                 [kitti_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/kitti/kitti_infos_val.pkl)                                                                 |                                                                        [kitti_infos_test](https://download.openmmlab.com/mmdetection3d/data/kitti/kitti_infos_test.pkl)                                                                         |
|                                                 NuScenes                                                  | [nuscenes_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl) [nuscenes_mini_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_mini_infos_train.pkl) | [nuscenes_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl)  [nuscenes_mini_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_mini_infos_val.pkl) |                                                                                                                                                                                                                                                 |
|                                                   Waymo                                                   |                                                         [waymo_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_infos_train.pkl)                                                          |                                                        [waymo_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_infos_val.pkl)                                                        | [waymo_infos_test.pkl](https://download.openmmlab.com/mmdetection3d/data/waymo/waymo_infos_test.pkl)   [waymo_infos_test_cam_only.pkl](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_infos_test_cam_only.pkl) |
| [Waymo-mini](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_mini.tar.gz) |                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                    |                                                                                                                                                                                                                                                 |
|                                                 SUN RGB-D                                                 |                                                               [sunrgbd_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/sunrgbd/sunrgbd_infos_train.pkl)                                                                |                                                              [sunrgbd_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/sunrgbd/sunrgbd_infos_val.pkl)                                                              |                                                                                                                                                                                                                                                 |
|                                                  ScanNet                                                  |                                                               [scannet_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/scannet/scannet_infos_train.pkl)                                                                |                                                              [scannet_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/scannet/scannet_infos_val.pkl)                                                              |                                                                   [scannet_infos_test.pkl](https://download.openmmlab.com/mmdetection3d/data/scannet/scannet_infos_test.pkl)                                                                    |
|                                               SemanticKitti                                               |                                                      [semantickitti_infos_train.pkl](https://download.openmmlab.com/mmdetection3d/data/semantickitti/semantickitti_infos_train.pkl)                                                       |                                                     [semantickitti_infos_val.pkl](https://download.openmmlab.com/mmdetection3d/data/semantickitti/semantickitti_infos_val.pkl)                                                     |                                                          [semantickitti_infos_test.pkl](https://download.openmmlab.com/mmdetection3d/data/semantickitti/semantickitti_infos_test.pkl)                                                           |
