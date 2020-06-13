### Prepare ScanNet Data
We follow the procedure in [votenet](https://github.com/facebookresearch/votenet/).

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the 'scans' folder to this level of directory.

2. In this level of directory, extract point clouds and annotations by running `python batch_load_scannet_data.py`.

3. Enter the project root directory, generate training data by running `python tools/create_data.py scannet --root-path ./data/scannet --out-dir ./data/scannet --extra-tag scannet`.

```
scannet
├── scannet_utils.py
├── load_scannet_data.py
├── README.md
├── scans

```
