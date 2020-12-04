## Installation

### Requirements

- Linux or macOS (Windows is not currently officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv)


### Install mmdetection

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

```python
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

```python
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

If you build PyTorch from source instead of installing the prebuilt pacakge,
you can use more CUDA versions such as 9.0.

c. Install [MMCV](https://mmcv.readthedocs.io/en/latest/).
*mmcv-full* is necessary since MMDetection3D relies on MMDetection, CUDA ops in *mmcv-full* are required.

The pre-build *mmcv-full* could be installed by running: (available versions could be found [here](https://mmcv.readthedocs.io/en/latest/#install-with-pip))

```shell
pip install mmcv-full==latest+torch1.5.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
```

Optionally, you could also build the full version from source:

```shell
pip install mmcv-full
```

d. Install [MMDetection](https://github.com/open-mmlab/mmdetection).

```shell
pip install git+https://github.com/open-mmlab/mmdetection.git
```

Optionally, you could also build MMDetection from source in case you want to modify the code:

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

**Important**:

1. The required versions of MMCV and MMDetection for different versions of MMDetection3D are as below. Please install the correct version of MMCV and MMDetection to avoid installation issues.

| MMDetection3D version | MMDetection version |    MMCV version     |
|:-------------------:|:-------------------:|:-------------------:|
| master              | mmdet>=2.5.0        | mmcv-full>=1.1.5, <=1.3|
| 0.8.0               | mmdet>=2.5.0        | mmcv-full>=1.1.5, <=1.3|
| 0.7.0               | mmdet>=2.5.0        | mmcv-full>=1.1.5, <=1.3|
| 0.6.0               | mmdet>=2.4.0        | mmcv-full>=1.1.3, <=1.2|
| 0.5.0               | 2.3.0               | mmcv-full==1.0.5|


e. Clone the MMDetection3D repository.

```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
```

f.Install build requirements and then install MMDetection3D.

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

2. Following the above instructions, mmdetection is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

5. The code can not be built for CPU only environment (where CUDA isn't available) for now.

### Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmdetection3d/blob/master/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmdetection3d docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection3d/data mmdetection3d
```

### A from-scratch setup script

Here is a full script for setting up mmdetection with conda.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y

# install mmcv
pip install mmcv-full

# install mmdetection
pip install git+https://github.com/open-mmlab/mmdetection.git

# install mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e .
```

### Using multiple MMDetection3D versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMDetection3D in the current directory.

To use the default MMDetection3D installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
