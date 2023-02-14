## 开始你的第一步

## 依赖

在本节中，我们将展示如何使用 PyTorch 准备环境。

MMDetection3D 支持在 Linux，Windows（实验性支持），MacOS 上运行，它需要 Python 3.7 以上，CUDA 9.2 以上和 PyTorch 1.6 以上。

```{note}
如果您对 PyTorch 有经验并且已经安装了它，您可以直接跳转到[下一小节](#安装流程)。否则，您可以按照下述步骤进行准备。
```

**步骤 0.** 从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda。

**步骤 1.** 创建并激活一个 conda 环境。

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**步骤 2.** 基于 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装 PyTorch，例如：

在 GPU 平台上：

```shell
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上：

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## 安装流程

我们推荐用户参照我们的最佳实践安装 MMDetection3D。不过，整个过程也是可定制化的，更多信息请参考[自定义安装](#自定义安装)章节。

### 最佳实践

**步骤 0.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine)，[MMCV](https://github.com/open-mmlab/mmcv) 和 [MMDetection](https://github.com/open-mmlab/mmdetection)。

```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc0'
mim install 'mmdet>=3.0.0rc0'
```

**注意**：在 MMCV-v2.x 中，`mmcv-full` 改名为 `mmcv`，如果您想安装不包含 CUDA 算子的 `mmcv`，您可以使用 `mim install "mmcv-lite>=2.0.0rc1"` 安装精简版。

**步骤 1.** 安装 MMDetection3D。

方案 a：如果您开发并直接运行 mmdet3d，从源码安装它：

```shell
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
# "-b dev-1.x" 表示切换到 `dev-1.x` 分支。
cd mmdetection3d
pip install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

方案 b：如果您将 mmdet3d 作为依赖或第三方 Python 包使用，使用 MIM 安装：

```shell
mim install "mmdet3d>=1.1.0rc0"
```

注意：

1. 如果您希望使用 `opencv-python-headless` 而不是 `opencv-python`，您可以在安装 MMCV 之前安装它。

2. 一些安装依赖是可选的。简单地运行 `pip install -v -e .` 将会安装最低运行要求的版本。如果想要使用一些可选依赖项，例如 `albumentations` 和 `imagecorruptions`，可以使用 `pip install -r requirements/optional.txt` 进行手动安装，或者在使用 `pip` 时指定所需的附加功能（例如 `pip install -v -e .[optional]`），支持附加功能的有效键值包括 `all`、`tests`、`build` 以及 `optional`。

   我们已经支持 `spconv 2.0`。如果用户已经安装 `spconv 2.0`，代码会默认使用 `spconv 2.0`，它会比原生 `mmcv spconv` 使用更少的 GPU 内存。用户可以使用下列的命令来安装 `spconv 2.0`：

   ```shell
   pip install cumm-cuxxx
   pip install spconv-cuxxx
   ```

   `xxx` 表示环境中的 CUDA 版本。

   例如，使用 CUDA 10.2，对应命令是 `pip install cumm-cu102 && pip install spconv-cu102`。

   支持的 CUDA 版本包括 10.2，11.1，11.3 和 11.4。用户也可以通过源码编译来安装。更多细节请参考[spconv v2.x](https://github.com/traveller59/spconv)。

   我们也支持 `Minkowski Engine` 作为稀疏卷积的后端。如果需要，请参考[安装指南](https://github.com/NVIDIA/MinkowskiEngine#installation) 或者使用 `pip` 来安装：

   ```shell
   conda install openblas-devel -c anaconda
   pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=/opt/conda/include" --install-option="--blas=openblas"
   ```

3. 我们的代码目前不能在只有 CPU 的环境（CUDA 不可用）下编译。

### 验证安装

为了验证 MMDetection3D 是否安装正确，我们提供了一些示例代码来执行模型推理。

**步骤 1.** 我们需要下载配置文件和模型权重文件。

```shell
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .
```

下载将需要几秒钟或更长时间，这取决于您的网络环境。完成后，您会在当前文件夹中发现两个文件 `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py` 和 `hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth`。

**步骤 2.** 推理验证。

方案 a：如果您从源码安装 MMDetection3D，那么直接运行以下命令进行验证：

```shell
python demo/pcd_demo.py demo/data/kitti/000008.bin pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show
```

您会看到一个带有点云的可视化界面，其中包含有在汽车上绘制的检测框。

**注意**：

如果您想输入一个 `.ply` 文件，您可以使用如下函数将它转换成 `.bin` 格式。然后您可以使用转化的 `.bin` 文件来运行样例。请注意在使用此脚本之前，您需要安装 `pandas` 和 `plyfile`。这个函数也可以用于训练 `ply 数据`时作为数据预处理来使用。

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

如果您有其他格式的点云数据（`.off`，`.obj` 等），您可以使用 `trimesh` 将它们转化成 `.ply`。

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

方案 b：如果您使用 MIM 安装 MMDetection3D，那么可以打开您的 Python 解析器，复制并粘贴以下代码：

```python
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.utils import register_all_modules

register_all_modules()
config_file = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
checkpoint_file = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
model = init_model(config_file, checkpoint_file)
inference_detector(model, 'demo/data/kitti/000008.bin')
```

您将会看到一个包含 `Det3DDataSample` 的列表，预测结果在 `pred_instances_3d` 里面，包含有检测框，类别和得分。

### 自定义安装

#### CUDA 版本

在安装 PyTorch 时，您需要指定 CUDA 的版本。如果您不清楚应该选择哪一个，请遵循我们的建议：

- 对于 Ampere 架构的 NVIDIA GPU，例如 GeForce 30 系列以及 NVIDIA A100，CUDA 11 是必需的。
- 对于更早的 NVIDIA GPU，CUDA 11 是向后兼容的，但 CUDA 10.2 提供更好的兼容性，并且更轻量。

请确保 GPU 驱动版本满足最低的版本需求。更多信息请参考此[表格](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

```{note}
如果您遵循我们的最佳实践，您只需要安装 CUDA 运行库，这是因为不需要在本地编译 CUDA 代码。但如果您希望从源码编译 MMCV，或者开发其他 CUDA 算子，那么您需要从 NVIDIA 的[官网](https://developer.nvidia.com/cuda-downloads)安装完整的 CUDA 工具链，并且该版本应该与 PyTorch 的 CUDA 版本相匹配，比如在 `conda install` 指令里指定 cudatoolkit 版本。
```

#### 不通过 MIM 安装 MMEngine

如果想要使用 pip 而不是 MIM 安装 MMEngine，请参考 [MMEngine 安装指南](https://mmengine.readthedocs.io/zh_CN/latest/get_started/installation.html)。

例如，您可以通过以下指令安装 MMEngine：

```shell
pip install mmengine
```

#### 不通过 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 拓展，因此其对 PyTorch 的依赖更复杂。MIM 会自动解决此类依赖关系并使安装更容易。但这不是必需的。

如果想要使用 pip 而不是 MIM 安装 MMCV，请参考 [MMCV 安装指南](https://mmcv.readthedocs.io/zh_CN/2.x/get_started/installation.html)。这需要用指定 url 的形式手动指定对应的 PyTorch 和 CUDA 版本。

例如，下述指令将会安装基于 PyTorch 1.12.x 和 CUDA 11.6 编译的 MMCV：

```shell
pip install "mmcv>=2.0.0rc1" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

#### 在 Google Colab 中安装

[Google Colab](https://colab.research.google.com/) 通常已经安装了 PyTorch，因此我们只需要用如下命令安装 MMEngine，MMCV，MMDetection 和 MMDetection3D 即可。

**步骤 1.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine)，[MMCV](https://github.com/open-mmlab/mmcv) 和 [MMDetection](https://github.com/open-mmlab/mmdetection)。

```shell
!pip3 install openmim
!mim install mmengine
!mim install "mmcv>=2.0.0rc1,<2.1.0"
!mim install "mmdet>=3.0.0rc0,<3.1.0"
```

**步骤 2.** 从源码安装 MMDetection3D。

```shell
!git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
%cd mmdetection3d
!pip install -e .
```

**步骤 3.** 验证安装是否成功。

```python
import mmdet3d
print(mmdet3d.__version__)
# 预期输出：1.1.0rc0 或其它版本号。
```

```{note}
在 Jupyter Notebook 中，感叹号 `!` 用于执行外部命令，而 `%cd` 是一个[魔术命令](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd)，用于切换 Python 的工作路径。
```

#### 通过 Docker 使用 MMDetection3D

我们提供了 [Dockerfile](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docker/Dockerfile) 来构建一个镜像。请确保您的 [docker 版本](https://docs.docker.com/engine/install/) >= 19.03。

```shell
# 基于 PyTorch 1.6，CUDA 10.1 构建镜像
# 如果您想要其他版本，只需要修改 Dockerfile
docker build -t mmdetection3d docker/
```

用以下命令运行 Docker 镜像：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection3d/data mmdetection3d
```

### 故障排除

如果您在安装过程中遇到一些问题，请先参考 [FAQ](notes/faq.md) 页面。如果没有找到对应的解决方案，您也可以在 GitHub [提一个问题](https://github.com/open-mmlab/mmdetection3d/issues/new/choose)。

### 使用多个 MMDetection3D 版本进行开发

训练和测试的脚本已经在 `PYTHONPATH` 中进行了修改，以确保脚本使用当前目录中的 MMDetection3D。

要使环境中安装默认版本的 MMDetection3D 而不是当前正在使用的，可以删除出现在相关脚本中的代码：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
