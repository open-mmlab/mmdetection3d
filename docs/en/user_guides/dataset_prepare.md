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

```

## Download and Data Preparation

### KITTI

Download KITTI 3D detection data [HERE](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Prepare KITTI data splits by running:

```bash
mkdir ./data/kitti/ && mkdir ./data/kitti/ImageSets

# Download data split
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt
```

Then generate info files by running:

```bash
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

In an environment using slurm, users may run the following command instead:

```bash
sh tools/create_data.sh <partition> kitti
```

### Waymo

Download Waymo open dataset V1.2 [HERE](https://waymo.com/open/download/) and its data split [HERE](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o?usp=sharing). Then put `.tfrecord` files into corresponding folders in `data/waymo/waymo_format/` and put the data split `.txt` files into `data/waymo/kitti_format/ImageSets`. Download ground truth `.bin` file for validation set [HERE](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0/validation/ground_truth_objects) and put it into `data/waymo/waymo_format/`. A tip is that you can use `gsutil` to download the large-scale dataset with commands. You can take this [tool](https://github.com/RalphMao/Waymo-Dataset-Tool) as an example for more details. Subsequently, prepare waymo data by running:

```bash
python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --workers 128 --extra-tag waymo
```

Note that:

- If your local disk does not have enough space for saving converted data, you can change the `--out-dir` to anywhere else. Just remember to create folders and prepare data there in advance and link them back to `data/waymo/kitti_format` after the data conversion.

- If you want faster evaluation on Waymo, you can download the preprocessed [metainfo](https://download.openmmlab.com/mmdetection3d/data/waymo/idx2metainfo.pkl) containing `contextname` and `timestamp` to the directory `data/waymo/waymo_format/`. Then, the dataset config is modified like the following:

  ```python
  val_evaluator = dict(
      type='WaymoMetric',
      ann_file='./data/waymo/kitti_format/waymo_infos_val.pkl',
      waymo_bin_file='./data/waymo/waymo_format/gt.bin',
      data_root='./data/waymo/waymo_format',
      file_client_args=file_client_args,
      convert_kitti_format=True,
      idx2metainfo='data/waymo/waymo_format/idx2metainfo.pkl'
      )
  ```

  Now, this trick is only used for LiDAR-based detection methods.

### NuScenes

Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

### Lyft

Download Lyft 3D detection data [HERE](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data). Prepare Lyft data by running:

```bash
python tools/create_data.py lyft --root-path ./data/lyft --out-dir ./data/lyft --extra-tag lyft --version v1.01
python tools/dataset_converters/lyft_data_fixer.py --version v1.01 --root-folder ./data/lyft
```

Note that we follow the original folder names for clear organization. Please rename the raw folders as shown above. Also note that the second command serves the purpose of fixing a corrupted lidar data file. Please refer to the [discussion](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/110000) for more details.

### S3DIS, ScanNet and SUN RGB-D

To prepare S3DIS data, please see its [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/s3dis/README.md).

To prepare ScanNet data, please see its [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/scannet/README.md).

To prepare SUN RGB-D data, please see its [README](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/data/sunrgbd/README.md).

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
