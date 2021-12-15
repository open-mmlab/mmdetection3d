### Prepare ScanNet Data for Indoor Detection or Segmentation Task

We follow the procedure in [votenet](https://github.com/facebookresearch/votenet/).

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the 'scans' folder to this level of directory. If you are performing segmentation tasks and want to upload the results to its official [benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/), please also link or move the 'scans_test' folder to this directory.

2. In this directory, extract point clouds and annotations by running `python batch_load_scannet_data.py`. Add the `--max_num_point 50000` flag if you only use the ScanNet data for the detection task. It will downsample the scenes to less points.

3. In this directory, extract RGB image with poses by running `python extract_posed_images.py`. This step is optional. Skip it if you don't plan to use multi-view RGB images. Add `--max-images-per-scene -1` to disable limiting number of images per scene. ScanNet scenes contain up to 5000+ frames per each. After extraction, all the .jpg images require 2 Tb disk space. The recommended 300 images per scene require less then 100 Gb. For example multi-view 3d detector ImVoxelNet samples 50 and 100 images per training and test scene.

4. Enter the project root directory, generate training data by running

```bash
python tools/create_data.py scannet --root-path ./data/scannet --out-dir ./data/scannet --extra-tag scannet
```

The overall process could be achieved through the following script

```bash
python batch_load_scannet_data.py
python extract_posed_images.py
cd ../..
python tools/create_data.py scannet --root-path ./data/scannet --out-dir ./data/scannet --extra-tag scannet
```

The directory structure after pre-processing should be as below

```
scannet
├── meta_data
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
├── posed_images
│   ├── scenexxxx_xx
│   │   ├── xxxxxx.txt
│   │   ├── xxxxxx.jpg
│   │   ├── intrinsic.txt
├── scannet_infos_train.pkl
├── scannet_infos_val.pkl
├── scannet_infos_test.pkl

```
