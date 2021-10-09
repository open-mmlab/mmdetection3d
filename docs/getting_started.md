# Prerequisites

- Linux or macOS (Windows is not currently officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)


The required versions of MMCV, MMDetection and MMSegmentation for different versions of MMDetection3D are as below. Please install the correct version of MMCV, MMDetection and MMSegmentation to avoid installation issues.

| MMDetection3D version | MMDetection version | MMSegmentation version |    MMCV version     |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
| master              | mmdet>=2.14.0, <=3.0.0| mmseg>=0.14.1, <=1.0.0 | mmcv-full>=1.3.8, <=1.4|
| 0.17.1              | mmdet>=2.14.0, <=3.0.0| mmseg>=0.14.1, <=1.0.0 | mmcv-full>=1.3.8, <=1.4|
| 0.17.0              | mmdet>=2.14.0, <=3.0.0| mmseg>=0.14.1, <=1.0.0 | mmcv-full>=1.3.8, <=1.4|
| 0.16.0              | mmdet>=2.14.0, <=3.0.0| mmseg>=0.14.1, <=1.0.0 | mmcv-full>=1.3.8, <=1.4|
| 0.15.0              | mmdet>=2.14.0, <=3.0.0| mmseg>=0.14.1, <=1.0.0 | mmcv-full>=1.3.8, <=1.4|
| 0.14.0              | mmdet>=2.10.0, <=2.11.0| mmseg==0.14.0 | mmcv-full>=1.3.1, <=1.4|
| 0.13.0              | mmdet>=2.10.0, <=2.11.0| Not required  | mmcv-full>=1.2.4, <=1.4|
| 0.12.0              | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.2.4, <=1.4|
| 0.11.0              | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.2.4, <=1.4|
| 0.10.0              | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.2.4, <=1.4|
| 0.9.0               | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.2.4, <=1.4|
| 0.8.0               | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.1.5, <=1.4|
| 0.7.0               | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.1.5, <=1.4|
| 0.6.0               | mmdet>=2.4.0, <=2.11.0 | Not required  | mmcv-full>=1.1.3, <=1.2|
| 0.5.0               | 2.3.0                  | Not required  | mmcv-full==1.0.5|

# Installation

## Install MMDetection3D

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g. 1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

```python
conda install pytorch==1.5.0 cudatoolkit=10.1 torchvision==0.6.0 -c pytorch
```

`E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

```python
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

If you build PyTorch from source instead of installing the prebuilt pacakge,
you can use more CUDA versions such as 9.0.

**c. Install [MMCV](https://mmcv.readthedocs.io/en/latest/).**
*mmcv-full* is necessary since MMDetection3D relies on MMDetection, CUDA ops in *mmcv-full* are required.

`e.g.` The pre-build *mmcv-full* could be installed by running: (available versions could be found [here](https://mmcv.readthedocs.io/en/latest/#install-with-pip))

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the latest `mmcv-full` with `CUDA 11` and `PyTorch 1.7.0`, use the following command:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
Optionally, you could also build the full version from source:

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
cd ..
```

Or directly run

```shell
pip install mmcv-full
```

**d. Install [MMDetection](https://github.com/open-mmlab/mmdetection).**

```shell
pip install mmdet==2.14.0
```

Optionally, you could also build MMDetection from source in case you want to modify the code:

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.14.0  # switch to v2.14.0 branch
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

**e. Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).**

```shell
pip install mmsegmentation==0.14.1
```

Optionally, you could also build MMSegmentation from source in case you want to modify the code:

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v0.14.1  # switch to v0.14.1 branch
pip install -e .  # or "python setup.py develop"
```

**f. Clone the MMDetection3D repository.**

```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
```

**g.Install build requirements and then install MMDetection3D.**

```shell
pip install -v -e .  # or "python setup.py develop"
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

    > Important: Be sure to remove the `./build` folder if you reinstall mmdet with a different CUDA/PyTorch version.

    ```shell
    pip uninstall mmdet3d
    rm -rf ./build
    find . -name "*.so" | xargs rm
    ```

2. Following the above instructions, MMDetection3D is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

5. The code can not be built for CPU only environment (where CUDA isn't available) for now.

## Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmdetection3d/blob/master/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmdetection3d docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection3d/data mmdetection3d
```

## A from-scratch setup script

Here is a full script for setting up MMdetection3D with conda.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest PyTorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y

# install mmcv
pip install mmcv-full

# install mmdetection
pip install git+https://github.com/open-mmlab/mmdetection.git

# install mmsegmentation
pip install git+https://github.com/open-mmlab/mmsegmentation.git

# install mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e .
```

## Using multiple MMDetection3D versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMDetection3D in the current directory.

To use the default MMDetection3D installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

# Verification

## Verify with point cloud demo

We provide several demo scripts to test a single sample. Pre-trained models can be downloaded from [model zoo](model_zoo.md). To test a single-modality 3D detection on point cloud scenes:

```shell
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}]
```

Examples:

```shell
python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth
```

If you want to input a `ply` file, you can use the following function and convert it to `bin` format. Then you can use the converted `bin` file to generate demo.
Note that you need to install `pandas` and `plyfile` before using this script. This function can also be used for data preprocessing for training ```ply data```.

```python
import numpy as np
import pandas as pd
from plyfile import PlyData

def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # read file
    data = plydata.elements[0].data  # read data
    data_pd = pd.DataFrame(data)  # convert to DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # initialize array to store data
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(
            property_names):  # read data by property
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)
```

Examples:

```python
convert_ply('./test.ply', './test.bin')
```

If you have point clouds in other format (`off`, `obj`, etc.), you can use `trimesh` to convert them into `ply`.

```python
import trimesh

def to_ply(input_path, output_path, original_type):
    mesh = trimesh.load(input_path, file_type=original_type)  # read file
    mesh.export(output_path, file_type='ply')  # convert to ply
```

Examples:

```python
to_ply('./test.obj', './test.ply', 'obj')
```

More demos about single/multi-modality and indoor/outdoor 3D detection can be found in [demo](demo.md).

## High-level APIs for testing point clouds

### Synchronous interface

Here is an example of building the model and test given point clouds.

```python
from mmdet3d.apis import init_model, inference_detector

config_file = 'configs/votenet/votenet_8x8_scannet-3d-18class.py'
checkpoint_file = 'checkpoints/votenet_8x8_scannet-3d-18class_20200620_230238-2cea9c3a.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
point_cloud = 'test.bin'
result, data = inference_detector(model, point_cloud)
# visualize the results and save the results in 'results' folder
model.show_results(data, result, out_dir='results')
```
