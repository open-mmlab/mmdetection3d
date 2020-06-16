### Prepare ScanNet Data
We follow the procedure in [votenet](https://github.com/facebookresearch/votenet/).

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the 'scans' folder to this level of directory.

2. In this directory, extract point clouds and annotations by running `python batch_load_scannet_data.py`.

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
├── scannet_train_instance_data
├── points
├── instance_mask
├── semantic_mask
├── scannet_infos_train.pkl
├── scannet_infos_val.pkl

```
