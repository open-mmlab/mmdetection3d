### Prepare SUN RGB-D Data

We follow the procedure in [votenet](https://github.com/facebookresearch/votenet/).

1. Download SUNRGBD data [HERE](http://rgbd.cs.princeton.edu/data/). Then, move SUNRGBD.zip, SUNRGBDMeta2DBB_v2.mat, SUNRGBDMeta3DBB_v2.mat and SUNRGBDtoolbox.zip to the OFFICIAL_SUNRGBD folder, unzip the zip files.

2. Enter the `matlab` folder, Extract point clouds and annotations by running `extract_split.m`, `extract_rgbd_data_v2.m` and `extract_rgbd_data_v1.m`.

3. Enter the project root directory, Generate training data by running

```bash
python tools/create_data.py sunrgbd --root-path ./data/sunrgbd --out-dir ./data/sunrgbd --extra-tag sunrgbd
```

The overall process could be achieved through the following script

```bash
cd matlab
matlab -nosplash -nodesktop -r 'extract_split;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v2;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v1;quit;'
cd ../../..
python tools/create_data.py sunrgbd --root-path ./data/sunrgbd  --out-dir ./data/sunrgbd --extra-tag sunrgbd
```

NOTE: SUNRGBDtoolbox.zip should have MD5 hash `18d22e1761d36352f37232cba102f91f` (you can check the hash with `md5 SUNRGBDtoolbox.zip` on Mac OS or `md5sum SUNRGBDtoolbox.zip` on Linux)

NOTE: If you would like to play around with [ImVoteNet](../../configs/imvotenet/README.md), the image data (`./data/sunrgbd/sunrgbd_trainval/image`) are required. If you pre-processed the data before mmdet3d version 0.12.0, please pre-process the data again due to some updates in data pre-processing

```bash
python tools/create_data.py sunrgbd --root-path ./data/sunrgbd  --out-dir ./data/sunrgbd --extra-tag sunrgbd
```

The directory structure after pre-processing should be as below

```
sunrgbd
├── README.md
├── matlab
│   ├── extract_rgbd_data_v1.m
│   ├── extract_rgbd_data_v2.m
│   ├── extract_split.m
├── OFFICIAL_SUNRGBD
│   ├── SUNRGBD
│   ├── SUNRGBDMeta2DBB_v2.mat
│   ├── SUNRGBDMeta3DBB_v2.mat
│   ├── SUNRGBDtoolbox
├── sunrgbd_trainval
│   ├── calib
│   ├── depth
│   ├── image
│   ├── label
│   ├── label_v1
│   ├── seg_label
│   ├── train_data_idx.txt
│   ├── val_data_idx.txt
├── points
├── sunrgbd_infos_train.pkl
├── sunrgbd_infos_val.pkl

```
