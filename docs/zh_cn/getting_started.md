# 依赖

MMDetection3D 可以安装在 Linux, MacOS, (实验性支持 Windows) 的平台上，它具体需要下列安装包:

- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (如果你从源码编译 PyTorch, CUDA 9.0 也是兼容的。)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

```{note}
如果你已经装了 pytorch, 可以跳过这一部分，然后转到[下一章节](#安装). 如果没有，可以参照以下步骤安装环境。
```

**步骤 0.** 安装 MiniConda [官网](https://docs.conda.io/en/latest/miniconda.html).

**步骤 1.** 使用 conda 新建虚拟环境，并进入该虚拟环境.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**步骤 2.** 基于 [PyTorch 官网](https://pytorch.org/)安装 PyTorch 和 torchvision，例如：

GPU 环境下

```shell
conda install pytorch torchvision -c pytorch
```

CPU 环境下

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# 安装

我们建议用户参照我们的最佳实践 MMDetection3D。不过，整个过程也是可定制化的，具体可参照[自定义安装章节](#customize-installation)

## 最佳实践

如果你已经成功安装 CUDA 11.0，那么你可以使用这个快速安装命令进行 MMDetection3D 的安装。 否则，则参考下一小节的详细安装流程。

```shell
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -e .
```

**步骤 0. 通过[MIM](https://github.com/open-mmlab/mim) 安装  [MMCV](https://github.com/open-mmlab/mmcv).**

**步骤 1. 安装 [MMDetection](https://github.com/open-mmlab/mmdetection).**

```shell
pip install mmdet
```

同时，如果你想修改这部分的代码，也可以通过以下命令从源码编译 MMDetection：

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.24.0  # switch to v2.24.0 branch
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

**步骤 2. 安装 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).**

```shell
pip install mmsegmentation
```

同时，如果你想修改这部分的代码，也可以通过以下命令从源码编译 MMSegmentation：

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v0.20.0  # switch to v0.20.0 branch
pip install -e .  # or "python setup.py develop"
```

**步骤 3. 克隆 MMDetection3D 代码仓库.**

```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
```

**步骤 4. 安装依赖包和 MMDetection3D.**

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

4. 一些安装依赖是可以选择的。例如只需要安装最低运行要求的版本，则可以使用 `pip install -v -e .` 命令。如果希望使用可选择的像 `albumentations` 和 `imagecorruptions` 这种依赖项，可以使用 `pip install -r requirements/optional.txt ` 进行手动安装，或者在使用 `pip` 时指定所需的附加功能（例如 `pip install -v -e .[optional]`），支持附加功能的有效键值包括  `all`、`tests`、`build` 以及 `optional` 。

   我们已经支持 spconv2.0. 如果用户已经安装 spconv 2.0， 代码会默认使用 spconv 2.0。它可以比原生 mmcv spconv 使用更少的内存。 用户可以使用下列的命令来安装 spconv 2.0.

   ```bash
   pip install cumm-cuxxx
   pip install spconv-cuxxx
   ```

   xxx 表示  CUDA 的版本。

   例如, 使用 CUDA 10.2, 对应命令是  `pip install cumm-cu102 && pip install spconv-cu102`.

   支持的 CUDA 版本包括 10.2, 11.1, 11.3, and 11.4. 用户可以通过源码编译来在这些版本上安装. 具体细节请参考 [spconv v2.x](https://github.com/traveller59/spconv).

   我们同时也支持 Minkowski Engine 来作为稀疏卷机的后端. 如果需要，可以参照 [安装指南](https://github.com/NVIDIA/MinkowskiEngine#installation) 或使用 `pip`:

   ```shell
   conda install openblas-devel -c anaconda
   pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=/opt/conda/include" --install-option="--blas=openblas"
   ```

5. 我们的代码目前不能在只有 CPU 的环境（CUDA 不可用）下编译运行。

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

请注意在使用此脚本前，你需要先安装 `pandas` 和 `plyfile`。 这个函数也可使用在数据预处理当中，为了能够直接训练 `ply data`。

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

## 自定义安装

### CUDA 版本

当安装 PyTorch 的时候，你需要去指定 CUDA 的版本。如果你不清楚如何选择 CUDA 的版本，可以参考我们如下的建议：

- 对于 Ampere 的 NVIDIA GPU, 比如 GeForce 30 series 和 NVIDIA A100, CUDA 11 是必须的。
- 对于老款的 NVIDIA GPUs, CUDA 11 是可编译的，但是 CUDA 10.2 提供更好的可编译性，并且更轻量。

请确保GPU 驱动版本大于最低需求。这个[表格](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) 提供更多的信息。

```{note}
如果你参照最佳实践，你只需要安装 CUDA runtime libraries。 这是因为没有代码需要在本地通过 CUDA 编译。然而如果你需要编译MMCV源码，或者编译其他 CUDA 代码，你需要基于 NVIDIA [website](https://developer.nvidia.com/cuda-downloads) 安装完整的 CUDA toolkit，并且要保证它的版本跟 PyTorch 匹配。比如在 'conda install` 里对应的 cudatoolkit 版本。
```

### 不通过MIM 安装MMCV

MMCV 包含一些 C++ 和 CUDA 扩展,因此以复杂的方式依赖于 PyTorch。 MIM 会自动解决此类依赖关系并使安装更容易。但是，这不是必须的。

如果想要使用 pip 而不是 MIM 安装 MMCV, 请参考 [MMCV 安装指南](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). 这需要根据 PyTorch 版本及其 CUDA 版本手动指定 find-url。

例如, 下面的脚本安装 的 mmcv-full 是对应的 PyTorch 1.10.x 和 CUDA 11.3.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### 通过Docker 安装

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

## 故障排除

如果在安装过程中遇到什么问题，可以先参考 [FAQ](faq.md) 页面.
如果没有找到对应的解决方案，你也可以在 Github [提一个 issue](https://github.com/open-mmlab/mmdetection3d/issues/new/choose)。
