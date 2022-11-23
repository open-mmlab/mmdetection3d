# 依赖

在本节中，我们将展示如何使用 PyTorch 准备环境。MMDetection3D 可以安装在 Linux, MacOS,（实验性支持 Windows）的平台上，它具体需要下列安装包:

- Python 3.6+
- PyTorch 1.6+
- CUDA 9.2+（如果你从源码编译 PyTorch, CUDA 9.0 也是兼容的。）
- GCC 5+
- [MMEngine](https://mmengine.readthedocs.io/zh_CN/latest/#installation)
- [MMCV](https://mmcv.readthedocs.io/zh_CN/latest/#installation)

```{note}
如果你已经装了 pytorch，可以跳过这一部分，然后转到[下一章节](#安装)。如果没有，可以参照以下步骤安装环境。
```

**步骤 0.** 从[官网](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda。

**步骤 1.** 使用 conda 新建虚拟环境，并进入该虚拟环境。

```shell
# 鉴于 waymo-open-dataset-tf-2-6-0 要求 python>=3.7，我们推荐安装 python3.8
# 如果您想要安装 python<3.7，之后须确保安装 waymo-open-dataset-tf-2-x-0 (x<=4)
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**步骤 2.** 基于 [PyTorch 官网](https://pytorch.org/)安装 PyTorch，例如：

GPU 环境下：

```shell
conda install pytorch torchvision -c pytorch
```

CPU 环境下：

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# 安装

我们建议用户参照我们的最佳实践 MMDetection3D。不过，整个过程也是可定制化的，具体可参照[自定义安装章节](#%E8%87%AA%E5%AE%9A%E4%B9%89%E5%AE%89%E8%A3%85)

## 最佳实践

如果你已经成功安装 CUDA 11.0，那么你可以使用这个快速安装命令进行 MMDetection3D 的安装。否则，则参考下一小节的详细安装流程。

```shell
pip install openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc0'
mim install 'mmdet>=3.0.0rc0'
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
cd mmdetection3d
pip install -e .
```

**步骤 0.** 通过 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv)。

```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc0'
```

**步骤 1.** 安装 [MMDetection](https://github.com/open-mmlab/mmdetection)。

```shell
mim install 'mmdet>=3.0.0rc0'
```

同时，如果你想修改这部分的代码，也可以通过以下命令从源码编译 MMDetection：

```shell
git clone https://github.com/open-mmlab/mmdetection.git -b dev-3.x
# "-b dev-3.x" 表示切换到 `dev-3.x` 分支。
cd mmdetection
pip install -v -e .
# "-v" 表示更详细的信息输出
# "-e" 表示以可编辑的模式安装项目
# 因此本地对代码做的任何修改都会生效，而无需重新安装。
```

**步骤 2.** 克隆 MMDetection3D 代码仓库。

```shell
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
# "-b dev-1.x" 表示切换到 `dev-1.x` 分支。
cd mmdetection3d
```

**步骤 4.** 安装依赖包和 MMDetection3D。

```shell
pip install -v -e .  # 或者 "python setup.py develop"
```

注意：

1. Git 的 commit id 在步骤 d 将会被写入到版本号当中，例 0.6.0+2e7045c。版本号将保存在训练的模型里。推荐在每一次执行步骤 4 时，从 github 上获取最新的更新。如果基于 C++/CUDA 的代码被修改了，请执行以下步骤；

   > 重要: 如果你重装了不同版本的 CUDA 或者 PyTorch 的 mmdet，请务必移除 `./build` 文件。

   ```shell
   pip uninstall mmdet3d
   rm -rf ./build
   find . -name "*.so" | xargs rm
   ```

2. 按照上述说明，MMDetection3D 安装在 `dev` 模式下，因此在本地对代码做的任何修改都会生效，无需重新安装；

3. 如果希望使用 `opencv-python-headless` 而不是 `opencv-python`，可以在安装 MMCV 之前安装；

4. 一些安装依赖是可以选择的。例如只需要安装最低运行要求的版本，则可以使用 `pip install -v -e .` 命令。如果希望使用可选择的像 `albumentations` 和 `imagecorruptions` 这种依赖项，可以使用 `pip install -r requirements/optional.txt` 进行手动安装，或者在使用 `pip` 时指定所需的附加功能（例如 `pip install -v -e .[optional]`），支持附加功能的有效键值包括 `all`、`tests`、`build` 以及 `optional`。

   我们已经支持 `spconv 2.0`。如果用户已经安装 `spconv 2.0`，代码会默认使用 `spconv 2.0`。它可以比原生 `mmcv spconv` 使用更少的内存。用户可以使用下列的命令来安装 `spconv 2.0`.

   ```bash
   pip install cumm-cuxxx
   pip install spconv-cuxxx
   ```

   `xxx` 表示 CUDA 的版本。

   例如，使用 CUDA 10.2, 对应命令是 `pip install cumm-cu102 && pip install spconv-cu102`。

   支持的 CUDA 版本包括 10.2，11.1，11.3 和 11.4。用户可以通过源码编译来在这些版本上安装。具体细节请参考 [spconv v2.x](https://github.com/traveller59/spconv)。

   我们同时也支持 `Minkowski Engine` 来作为稀疏卷积的后端。如果需要，可以参照[安装指南](https://github.com/NVIDIA/MinkowskiEngine#installation)或使用 `pip` 来安装：

   ```shell
   conda install openblas-devel -c anaconda
   pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=/opt/conda/include" --install-option="--blas=openblas"
   ```

5. 我们的代码目前不能在只有 CPU 的环境（CUDA 不可用）下编译运行。

## 验证

### 通过点云样例程序来验证

我们提供了一些样例脚本去测试单个样本，预训练的模型可以从[模型库](model_zoo.md)中下载. 运行如下命令可以去测试点云场景下一个单模态的 3D 检测算法。

```shell
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}]
```

例如：

```shell
python demo/pcd_demo.py demo/data/kitti/000008.bin configs/second/second_hv-secfpn_8xb6-80e_kitti-3d-car.py checkpoints/second_hv-secfpn_8xb6-80e_kitti-3d-car_20200620_230238-393f000c.pth
```

如果你想输入一个 `.ply` 格式的文件，你可以使用如下函数将它转换为 `.bin` 的文件格式。然后就可以使用转化成 `.bin` 格式的文件去运行样例程序。
请注意在使用此脚本前，你需要先安装 `pandas` 和 `plyfile`。这个函数也可使用在数据预处理当中，为了能够直接训练 `ply data`。

```python
import numpy as np
import pandas as pd
from plyfile import PlyData

def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # 读取文件
    data = plydata.elements[0].data  # 读取数据
    data_pd = pd.DataFrame(data)  # 转换成 DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化数组来存储数据
    property_names = data[0].dtype.names  # 读取属性名称
    for i, name in enumerate(
            property_names):  # 通过属性读取数据
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)
```

例如：

```python
convert_ply('./test.ply', './test.bin')
```

如果你有其他格式的点云文件 (例：`.off`，`.obj`)，你可以使用 `trimesh` 将它们转化成 `.ply`。

```python
import trimesh

def to_ply(input_path, output_path, original_type):
    mesh = trimesh.load(input_path, file_type=original_type)  # 读取文件
    mesh.export(output_path, file_type='ply')  # 转换成 ply
```

例如：

```python
to_ply('./test.obj', './test.ply', 'obj')
```

更多的关于单/多模态和室内/室外的 3D 检测的样例可以在[此](user_guides/inference.md)找到。

## 自定义安装

### CUDA 版本

当安装 PyTorch 的时候，你需要去指定 CUDA 的版本。如果你不清楚如何选择 CUDA 的版本，可以参考我们如下的建议：

- 对于 Ampere 的 NVIDIA GPU, 比如 GeForce 30 series 和 NVIDIA A100, CUDA 11 是必须的。
- 对于老款的 NVIDIA GPUs, CUDA 11 是可编译的，但是 CUDA 10.2 提供更好的可编译性，并且更轻量。

请确保 GPU 驱动版本大于最低需求。更多信息请参考此[表格](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

```{note}
如果你参照最佳实践，你只需要安装 CUDA runtime libraries。这是因为没有代码需要在本地通过 CUDA 编译。然而如果你需要编译 MMCV 源码，或者编译其他 CUDA 代码，你需要基于 NVIDIA [website](https://developer.nvidia.com/cuda-downloads) 安装完整的 CUDA toolkit，并且要保证它的版本跟 PyTorch 匹配。比如在 `conda install` 指令里指定 cudatoolkit 版本。
```

### 不通过 MIM 安装 MMEngine

如果想要使用 pip 而不是 MIM 安装 MMEngine, 请参考 [MMEngine 安装指南](https://mmengine.readthedocs.io/zh_CN/latest/get_started/installation.html)。

例如，你可以通过以下指令安装 MMEngine。

```shell
pip install mmengine
```

### 不通过 MIM 安装 MMCV

MMCV 包含一些 C++ 和 CUDA 扩展，因此以复杂的方式依赖于 PyTorch。MIM 会自动解决此类依赖关系并使安装更容易。但是，这不是必须的。

如果想要使用 pip 而不是 MIM 安装 MMCV，请参考 [MMCV 安装指南](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html)。这需要根据 PyTorch 版本及其 CUDA 版本手动指定 find-url。

例如，下面的脚本安装 的 mmcv 是对应的 PyTorch 1.10.x 和 CUDA 11.3。

```shell
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### 通过 Docker 安装 MMDetection3D

我们提供了 [Dockerfile](https://github.com/open-mmlab/mmdetection3d/blob/master/docker/Dockerfile) 来建立一个镜像。

```shell
# 基于 PyTorch 1.6, CUDA 10.1 生成 docker 的镜像
docker build -t mmdetection3d -f docker/Dockerfile .
```

运行命令：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection3d/data mmdetection3d
```

## 从零开始的安装脚本

以下是一个基于 conda 安装 MMdetection3D 的脚本

```shell
# 鉴于 waymo-open-dataset-tf-2-6-0 要求 python>=3.7，我们推荐安装 python3.8
# 如果您想要安装 python<3.7，之后须确保安装 waymo-open-dataset-tf-2-x-0 (x<=4)
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab

# 使用默认的预编译 CUDA 版本（通常是最新的）安装最新的 PyTorch
conda install -c pytorch pytorch torchvision -y

# 安装 mmengine and mmcv
pip install openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc0'

# 安装 mmdetection
mim install 'mmdet>=3.0.0rc0'

# 安装 mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
cd mmdetection3d
pip install -e .
```

## 故障排除

如果在安装过程中遇到什么问题，可以先参考 [FAQ](notes/faq.md) 页面。如果没有找到对应的解决方案，你也可以在 Github [提一个 issue](https://github.com/open-mmlab/mmdetection3d/issues/new/choose)。
