### Prepare S3DIS Data

We follow the procedure in [pointnet](https://github.com/charlesq34/pointnet).

1. Download S3DIS data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). Download the ```Stanford3dDataset_v1.2_Aligned_Version.zip``` file and unzip it. Link or move the folder to this level of directory.

2. In this directory, extract point clouds and annotations by running `python collect_indoor3d_data.py`.

3. Enter the project root directory, generate training data by running

```bash
python tools/create_data.py s3dis --root-path ./data/s3dis --out-dir ./data/s3dis --extra-tag s3dis
```

The overall process could be achieved through the following script

```bash
python collect_indoor3d_data.py
cd ../..
python tools/create_data.py s3dis --root-path ./data/s3dis --out-dir ./data/s3dis --extra-tag s3dis
```

The directory structure after pre-processing should be as below

```
s3dis
├── meta_data
├── indoor3d_util.py
├── collect_indoor3d_data.py
├── README.md
├── Stanford3dDataset_v1.2_Aligned_Version
├── s3dis_data
├── points
│   ├── xxxxx.bin
├── instance_mask
│   ├── xxxxx.bin
├── semantic_mask
│   ├── xxxxx.bin
├── seg_info
│   ├── Area_1_label_weight.npy
│   ├── Area_1_resampled_scene_idxs.npy
│   ├── Area_2_label_weight.npy
│   ├── Area_2_resampled_scene_idxs.npy
│   ├── Area_3_label_weight.npy
│   ├── Area_3_resampled_scene_idxs.npy
│   ├── Area_4_label_weight.npy
│   ├── Area_4_resampled_scene_idxs.npy
│   ├── Area_5_label_weight.npy
│   ├── Area_5_resampled_scene_idxs.npy
│   ├── Area_6_label_weight.npy
│   ├── Area_6_resampled_scene_idxs.npy
├── s3dis_infos_Area_1.pkl
├── s3dis_infos_Area_2.pkl
├── s3dis_infos_Area_3.pkl
├── s3dis_infos_Area_4.pkl
├── s3dis_infos_Area_5.pkl
├── s3dis_infos_Area_6.pkl

```
