### Prepare ScanNet Data for Indoor Detection or Segmentation Task

We follow the procedure in [votenet](https://github.com/facebookresearch/votenet/).

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the 'scans' folder to this level of directory. If you are performing segmentation tasks and want to upload the results to its official [benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/), please also link or move the 'scans_test' folder to this directory.

2. In this directory, extract point clouds and annotations by running `python batch_load_scannet_data.py`. Add the `--max_num_point 50000` flag if you only use the ScanNet data for the detection task. It will downsample the scenes to less points.

3. Enter the project root directory, generate training data by running

```bash
python tools/create_data.py scannet --root-path ./data/scannet --out-dir ./data/scannet --extra-tag scannet
```

The overall process could be achieved through the following script

```bash
python batch_load_scannet_data.py
cd ../..
python tools/create_data.py scannet --root-path ./data/scannet --out-dir ./data/scannet --extra-tag scannet
```

The directory structure after pre-processing should be as below

```
scannet
├── scannet_utils.py
├── batch_load_scannet_data.py
├── load_scannet_data.py
├── scannet_utils.py
├── README.md
├── scans
├── scans_test
├── scannet_instance_data
├── points
│   ├── xxxxx.bin
├── instance_mask
│   ├── xxxxx.bin
├── semantic_mask
│   ├── xxxxx.bin
├── seg_info
│   ├── train_label_weight.npy
│   ├── train_resampled_scene_idxs.npy
│   ├── val_label_weight.npy
│   ├── val_resampled_scene_idxs.npy
├── scannet_infos_train.pkl
├── scannet_infos_val.pkl
├── scannet_infos_test.pkl

```
