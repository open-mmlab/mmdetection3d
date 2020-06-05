### Prepare ScanNet Data

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Move/link the `scans` folder such that under `scans` there should be folders with names such as `scene0001_01`.

2. Extract point clouds and annotations (semantic seg, instance seg etc.) by running `python batch_load_scannet_data.py`, which will create a folder named `scannet_train_detection_data` here.
