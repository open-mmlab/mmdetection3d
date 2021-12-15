# 依赖

- Linux or macOS (Windows is not currently officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

| MMDetection3D version | MMDetection version | MMSegmentation version |    MMCV version     |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
| master              | mmdet>=2.14.0, <=3.0.0| mmseg>=0.14.1, <=1.0.0 | mmcv-full>=1.3.8, <=1.4|
| 0.17.2              | mmdet>=2.14.0, <=3.0.0| mmseg>=0.14.1, <=1.0.0 | mmcv-full>=1.3.8, <=1.4|
| 0.17.1              | mmdet>=2.14.0, <=3.0.0| mmseg>=0.14.1, <=1.0.0 | mmcv-full>=1.3.8, <=1.4|
| 0.17.0              | mmdet>=2.14.0, <=3.0.0| mmseg>=0.14.1, <=1.0.0 | mmcv-full>=1.3.8, <=1.4|
| 0.16.0              | mmdet>=2.14.0, <=3.0.0| mmseg>=0.14.1, <=1.0.0 | mmcv-full>=1.3.8, <=1.4|
| 0.15.0              | mmdet>=2.14.0, <=3.0.0| mmseg>=0.14.1, <=1.0.0 | mmcv-full>=1.3.8, <=1.4|
| 0.14.0              | mmdet>=2.10.0, <=2.11.0| mmseg>=0.14.0 | mmcv-full>=1.3.1, <=1.4|
| 0.13.0              | mmdet>=2.10.0, <=2.11.0| Not required  | mmcv-full>=1.2.4, <=1.4|
| 0.12.0              | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.2.4, <=1.4|
| 0.11.0              | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.2.4, <=1.4|
| 0.10.0              | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.2.4, <=1.4|
| 0.9.0               | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.2.4, <=1.4|
| 0.8.0               | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.1.5, <=1.4|
| 0.7.0               | mmdet>=2.5.0, <=2.11.0 | Not required  | mmcv-full>=1.1.5, <=1.4|
| 0.6.0               | mmdet>=2.4.0, <=2.11.0 | Not required  | mmcv-full>=1.1.3, <=1.2|
| 0.5.0               | 2.3.0                  | Not required  | mmcv-full==1.0.5|

# 安装

## MMdetection3D 安装流程

**a. 使用 conda 新建虚拟环境，并进入该虚拟环境。**

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

**b. 基于 [PyTorch 官网](https://pytorch.org/)安装 PyTorch 和 torchvision，例如：**

```shell
conda install pytorch torchvision -c pytorch
```

**注意**：需要确保 CUDA 的编译版本和运行版本匹配。可以在 [PyTorch 官网](https://pytorch.org/)查看预编译包所支持的 CUDA 版本。

`例 1` 例如在 `/usr/local/cuda` 下安装了 CUDA 10.1， 并想安装 PyTorch 1.5，则需要安装支持 CUDA 10.1 的预构建 PyTorch：

```shell
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`例 2` 例如在 `/usr/local/cuda` 下安装了 CUDA 9.2， 并想安装 PyTorch 1.3.1，则需要安装支持 CUDA 9.2  的预构建 PyTorch：

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

如果不是安装预构建的包，而是从源码中构建 PyTorch，则可以使用更多的 CUDA 版本，例如 CUDA 9.0。

**c. 安装 [MMCV](https://mmcv.readthedocs.io/en/latest/).**
需要安装 *mmcv-full*，因为 MMDetection3D 依赖 MMDetection 且需要 *mmcv-full* 中基于 CUDA 的程序。

`例` 可以使用下面命令安装预编译版本的 *mmcv-full* ：(可使用的版本在[这里](https://mmcv.readthedocs.io/en/latest/#install-with-pip)可以找到)

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
需要把命令行中的 `{cu_version}` 和 `{torch_version}` 替换成对应的版本。例如：在 CUDA 11 和 PyTorch 1.7.0 的环境下，可以使用下面命令安装最新版本的 MMCV：

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

请参考 [MMCV](https://mmcv.readthedocs.io/en/latest/#installation) 获取不同版本的 MMCV 所兼容的的不同的 PyTorch 和 CUDA 版本。同时，也可以通过以下命令行从源码编译 MMCV：

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # 安装好 mmcv-full
cd ..
```

或者，可以直接使用命令行安装：

```shell
pip install mmcv-full
```

**d. 安装 [MMDetection](https://github.com/open-mmlab/mmdetection).**

```shell
pip install mmdet==2.14.0
```

同时，如果你想修改这部分的代码，也可以通过以下命令从源码编译 MMDetection：

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.14.0  # 转到 v2.14.0 分支
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

**e. 安装 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).**

```shell
pip install mmsegmentation==0.14.1
```
同时，如果你想修改这部分的代码，也可以通过以下命令从源码编译 MMSegmentation：

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v0.14.1  # switch to v0.14.1 branch
pip install -e .  # or "python setup.py develop"
```

**g. 安装依赖包和 MMDetection3D.**

```shell
pip install -v -e .  # or "python setup.py develop"
```

**注意：**

1. Git 的 commit id 在步骤 d 将会被写入到版本号当中，例 0.6.0+2e7045c 。版本号将保存在训练的模型里。推荐在每一次执行步骤 d 时，从 github 上获取最新的更新。如果基于 C++/CUDA 的代码被修改了，请执行以下步骤；

    > 重要: 如果你重装了不同版本的 CUDA 或者 PyTorch 的 mmdet，请务必移除 `./build` 文件。

    ```shell
    pip uninstall mmdet3d
    rm -rf ./build
    find . -name "*.so" | xargs rm
    ```

2. 按照上述说明，MMDetection3D 安装在 `dev` 模式下，因此在本地对代码做的任何修改都会生效，无需重新安装；

3. 如果希望使用 `opencv-python-headless` 而不是 `opencv-python`， 可以在安装 MMCV 之前安装；

4.  一些安装依赖是可以选择的。例如只需要安装最低运行要求的版本，则可以使用 `pip install -v -e .` 命令。如果希望使用可选择的像 `albumentations` 和 `imagecorruptions` 这种依赖项，可以使用 `pip install -r requirements/optional.txt ` 进行手动安装，或者在使用 `pip` 时指定所需的附加功能（例如 `pip install -v -e .[optional]`），支持附加功能的有效键值包括  `all`、`tests`、`build` 以及 `optional` 。

5. 我们的代码目前不能在只有 CPU 的环境（CUDA 不可用）下编译运行。

## 另一种选择：Docker Image

我们提供了 [Dockerfile](https://github.com/open-mmlab/mmdetection3d/blob/master/docker/Dockerfile) 来建立一个镜像。

```shell
# 基于 PyTorch 1.6, CUDA 10.1 生成 docker 的镜像
docker build -t mmdetection3d docker/
```

运行命令：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection3d/data mmdetection3d
```

## 从零开始的安装脚本

以下是一个基于 conda 安装 MMdetection3D 的脚本

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# 安装基于环境中默认 CUDA 版本下最新的 PyTorch (通常使用最新版本)
conda install -c pytorch pytorch torchvision -y

# 安装 mmcv
pip install mmcv-full

# 安装 mmdetection
pip install git+https://github.com/open-mmlab/mmdetection.git

# 安装 mmsegmentation
pip install git+https://github.com/open-mmlab/mmsegmentation.git

# 安装 mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e .
```

## 使用多版本的 MMDetection3D

训练和测试的脚本已经在 PYTHONPATH 中进行了修改，以确保脚本使用当前目录中的 MMDetection3D。

要使环境中安装默认的 MMDetection3D 而不是当前正在在使用的，可以删除出现在相关脚本中的代码：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

# 验证

## 通过点云样例程序来验证

我们提供了一些样例脚本去测试单个样本，预训练的模型可以从[模型库](model_zoo.md)中下载. 运行如下命令可以去测试点云场景下一个单模态的 3D 检测算法。

```shell
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}]
```

例:

```shell
python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth
```

如果你想输入一个 `ply` 格式的文件，你可以使用如下函数将它转换为 `bin` 的文件格式。然后就可以使用转化成 `bin` 格式的文件去运行样例程序。

请注意在使用此脚本前，你需要先安装 `pandas` 和 `plyfile`。 这个函数也可使用在数据预处理当中，为了能够直接训练 ```ply data```。

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

例:

```python
convert_ply('./test.ply', './test.bin')
```

如果你有其他格式的点云文件 (例：`off`, `obj`), 你可以使用 `trimesh` 将它们转化成 `ply`.

```python
import trimesh

def to_ply(input_path, output_path, original_type):
    mesh = trimesh.load(input_path, file_type=original_type)  # read file
    mesh.export(output_path, file_type='ply')  # convert to ply
```

例:

```python
to_ply('./test.obj', './test.ply', 'obj')
```

更多的关于单/多模态和室内/室外的 3D 检测的样例可以在[此](demo.md)找到.

## 测试点云的高级接口

### 同步接口

这里有一个例子去说明如何构建模型以及测试给出的点云：

```python
from mmdet3d.apis import init_model, inference_detector

config_file = 'configs/votenet/votenet_8x8_scannet-3d-18class.py'
checkpoint_file = 'checkpoints/votenet_8x8_scannet-3d-18class_20200620_230238-2cea9c3a.pth'

# 从配置文件和预训练的模型文件中构建模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 测试单个文件并可视化结果
point_cloud = 'test.bin'
result, data = inference_detector(model, point_cloud)
# 可视化结果并且将结果保存到 'results' 文件夹
model.show_results(data, result, out_dir='results')
```
