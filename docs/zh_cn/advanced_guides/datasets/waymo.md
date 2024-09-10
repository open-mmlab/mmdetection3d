# Waymo 数据集

本文档页包含了关于 MMDetection3D 中 Waymo 数据集用法的教程。

## 数据集准备

在准备 Waymo 数据集之前，如果您之前只安装了 `requirements/build.txt` 和 `requirements/runtime.txt` 中的依赖，请通过运行如下指令额外安装 Waymo 数据集所依赖的官方包：

```
pip install waymo-open-dataset-tf-2-6-0
```

或者

```
pip install -r requirements/optional.txt
```

和准备数据集的通用方法一致，我们推荐将数据集根目录软链接至 `$MMDETECTION3D/data`。
由于原始 Waymo 数据的格式基于 `tfrecord`，我们需要将原始数据进行预处理，以便于训练和测试时使用。我们的方法是将它们转换为 KITTI 格式。

处理之前，文件目录结构组织如下：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── waymo
│   │   ├── waymo_format
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
│   │   │   ├── gt.bin
│   │   │   ├── cam_gt.bin
│   │   │   ├── fov_gt.bin
│   │   ├── kitti_format
│   │   │   ├── ImageSets

```

您可以在[这里](https://waymo.com/open/download/)下载 1.2 版本的 Waymo 公开数据集，并在[这里](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o?usp=sharing)下载其训练/验证/测试集拆分文件。接下来，请将 `tfrecord` 文件放入 `data/waymo/waymo_format/` 下的对应文件夹，并将 txt 格式的数据集拆分文件放入 `data/waymo/kitti_format/ImageSets`。在[这里](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0/validation/ground_truth_objects)下载验证集使用的 bin 格式真实标注 (Ground Truth) 文件并放入 `data/waymo/waymo_format/`。小窍门：您可以使用 `gsutil` 来在命令行下载大规模数据集。您可以将该[工具](https://github.com/RalphMao/Waymo-Dataset-Tool) 作为一个例子来查看更多细节。之后，通过运行如下指令准备 Waymo 数据：

```bash
# TF_CPP_MIN_LOG_LEVEL=3 will disable all logging output from TensorFlow.
# The number of `--workers` depends on the maximum number of cores in your CPU.
TF_CPP_MIN_LOG_LEVEL=3 python tools/create_data.py waymo --root-path ./data/waymo --out-dir ./data/waymo --workers 128 --extra-tag waymo --version v1.4
```

请注意，如果您的本地磁盘没有足够空间保存转换后的数据，您可以将 `--out-dir` 改为其他目录；只要在创建文件夹、准备数据并转换格式后，将数据文件链接到 `data/waymo/kitti_format` 即可。

在数据转换后，文件目录结构应组织如下：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── waymo
│   │   ├── waymo_format
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
│   │   │   ├── gt.bin
│   │   │   ├── cam_gt.bin
│   │   │   ├── fov_gt.bin
│   │   ├── kitti_format
│   │   │   ├── ImageSets
│   │   │   ├── training
│   │   │   │   ├── image_0
│   │   │   │   ├── image_1
│   │   │   │   ├── image_2
│   │   │   │   ├── image_3
│   │   │   │   ├── image_4
│   │   │   │   ├── velodyne
│   │   │   ├── testing
│   │   │   │   ├── (the same as training)
│   │   │   ├── waymo_gt_database
│   │   │   ├── waymo_infos_trainval.pkl
│   │   │   ├── waymo_infos_train.pkl
│   │   │   ├── waymo_infos_val.pkl
│   │   │   ├── waymo_infos_test.pkl
│   │   │   ├── waymo_dbinfos_train.pkl

```

- `kitti_format/training/image_{0-4}/{a}{bbb}{ccc}.jpg` 因为 Waymo 数据的来源包含数个相机，这里我们将每个相机对应的图像和标签文件分别存储，并将相机位姿 (pose) 文件存储下来以供后续处理连续多帧的点云。我们使用 `{a}{bbb}{ccc}` 的名称编码方式为每帧数据命名，其中 `a` 是不同数据拆分的前缀（`0` 指代训练集，`1` 指代验证集，`2` 指代测试集），`bbb` 是分割部分 (segment) 的索引，而 `ccc` 是帧索引。您可以轻而易举地按照如上命名规则定位到所需的帧。我们将训练和验证所需数据按 KITTI 的方式集合在一起，然后将训练集/验证集/测试集的索引存储在 `ImageSet` 下的文件中。
- `kitti_format/training/velodyne/{a}{bbb}{ccc}.bin` 当前样本的点云数据
- `kitti_format/waymo_gt_database/xxx_{Car/Pedestrian/Cyclist}_x.bin`. 训练数据集的每个 3D 包围框中包含的点云数据。这些点云会在数据增强中被使用，例如. `ObjectSample`. `xxx` 表示训练样本的索引，`x` 表示实例在当前样本中的索引。
- `kitti_format/waymo_infos_train.pkl`. 训练数据集，该字典包含了两个键值：`metainfo` 和 `data_list`。`metainfo` 包含数据集的基本信息，例如 `dataset`、`version` 和 `info_version`。`data_list` 是由字典组成的列表，每个字典（以下简称 `info`）包含了单个样本的所有详细信息。:
  - info\['sample_idx'\]: 样本在整个数据集的索引。
  - info\['ego2global'\]: 自车到全局坐标的变换矩阵。（4x4 列表）
  - info\['timestamp'\]：样本数据时间戳。
  - info\['context_name'\]: 语境名，表示样本从哪个 `*.tfrecord` 片段中提取的。
  - info\['lidar_points'\]: 是一个字典，包含了所有与激光雷达点相关的信息。
    - info\['lidar_points'\]\['lidar_path'\]: 激光雷达点云数据的文件名。
    - info\['lidar_points'\]\['num_pts_feats'\]: 点的特征维度。
  - info\['lidar_sweeps'\]: 是一个列表，包含了历史帧信息。
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['lidar_path'\]: 第 i 帧的激光雷达数据的文件路径。
    - info\['lidar_sweeps'\]\[i\]\['ego2global'\]: 第 i 帧的激光雷达传感器到自车的变换矩阵。（4x4 列表）
    - info\['lidar_sweeps'\]\[i\]\['timestamp'\]: 第 i 帧的样本数据时间戳。
  - info\['images'\]: 是一个字典，包含与每个相机对应的六个键值：`'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_SIDE_LEFT'`, `'CAM_SIDE_RIGHT'`。每个字典包含了对应相机的所有数据信息。
    - info\['images'\]\['CAM_XXX'\]\['img_path'\]: 图像的文件名。
    - info\['images'\]\['CAM_XXX'\]\['height'\]: 图像的高
    - info\['images'\]\['CAM_XXX'\]\['width'\]: 图像的宽
    - info\['images'\]\['CAM_XXX'\]\['cam2img'\]: 当 3D 点投影到图像平面时需要的内参信息相关的变换矩阵。（3x3 列表）
    - info\['images'\]\['CAM_XXX'\]\['lidar2cam'\]: 激光雷达传感器到该相机的变换矩阵。（4x4 列表）
    - info\['images'\]\['CAM_XXX'\]\['lidar2img'\]: 激光雷达传感器到图像平面的变换矩阵。（4x4 列表）
  - info\['image_sweeps'\]: 是一个列表，包含了历史帧信息。
    - info\['image_sweeps'\]\[i\]\['images'\]\['CAM_XXX'\]\['img_path'\]: 第i帧的图像的文件名.
    - info\['image_sweeps'\]\[i\]\['ego2global'\]: 第 i 帧的自车到全局坐标的变换矩阵。（4x4 列表）
    - info\['image_sweeps'\]\[i\]\['timestamp'\]: 第 i 帧的样本数据时间戳。
  - info\['instances'\]: 是一个字典组成的列表。每个字典包含单个实例的所有标注信息。对于其中的第 i 个实例，我们有：
    - info\['instances'\]\[i\]\['bbox_3d'\]: 长度为 7 的列表，以 (x, y, z, l, w, h, yaw) 的顺序表示实例的 3D 边界框。
    - info\['instances'\]\[i\]\['bbox'\]: 2D 边界框标注（，顺序为 \[x1, y1, x2, y2\] 的列表。有些实例可能没有对应的 2D 边界框标注。
    - info\['instances'\]\[i\]\['bbox_label_3d'\]: 整数表示实例的标签，-1 代表忽略。
    - info\['instances'\]\[i\]\['bbox_label'\]: 整数表示实例的标签，-1 代表忽略。
    - info\['instances'\]\[i\]\['num_lidar_pts'\]: 每个 3D 边界框内包含的激光雷达点数。
    - info\['instances'\]\[i\]\['camera_id'\]: 当前实例最可见相机的索引。
    - info\['instances'\]\[i\]\['group_id'\]: 当前实例在当前样本中的索引。
  - info\['cam_sync_instances'\]: 是一个字典组成的列表。每个字典包含单个实例的所有标注信息。它的形式与 \['instances'\]相同. 但是, \['cam_sync_instances'\] 专门用于基于多视角相机的三维目标检测任务。
  - info\['cam_instances'\]: 是一个字典，包含以下键值： `'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_SIDE_LEFT'`, `'CAM_SIDE_RIGHT'`. 对于基于视觉的 3D 目标检测任务，我们将整个场景的 3D 标注划分至它们所属于的相应相机中。对于其中的第 i 个实例，我们有：
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_3d'\]: 长度为 7 的列表，以 (x, y, z, l, h, w, yaw) 的顺序表示实例的 3D 边界框。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox'\]: 2D 边界框标注（3D 框投影的矩形框），顺序为 \[x1, y1, x2, y2\] 的列表。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_label_3d'\]: 实例标签。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_label'\]: 实例标签。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['center_2d'\]: 3D 框投影到图像上的中心点，大小为 (2, ) 的列表。
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['depth'\]: 3D 框投影中心的深度。

## 训练

考虑到原始数据集中的数据有很多相似的帧，我们基本上可以主要使用一个子集来训练我们的模型。在我们初步的基线中，我们在每五帧图片中加载一帧。得益于我们的超参数设置和数据增强方案，我们得到了比 Waymo [原论文](https://arxiv.org/pdf/1912.04838.pdf)中更好的性能。请移步 `configs/pointpillars/` 下的 README.md 以查看更多配置和性能相关的细节。我们会尽快发布一个更完整的 Waymo 基准榜单 (benchmark)。

## 评估

为了在 Waymo 数据集上进行检测性能评估，请按照[此处指示](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md)构建用于计算评估指标的二进制文件 `compute_detection_metrics_main`，并将它置于 `mmdet3d/core/evaluation/waymo_utils/` 下。您基本上可以按照下方命令安装 `bazel`，然后构建二进制文件：

```shell
# download the code and enter the base directory
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
# git clone https://github.com/Abyssaledge/waymo-open-dataset-master waymo-od # if you want to use faster multi-thread version.
cd waymo-od
git checkout remotes/origin/master

# use the Bazel build system
sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=3.1.0
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential

# configure .bazelrc
./configure.sh
# delete previous bazel outputs and reset internal caches
bazel clean

bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main ../mmdetection3d/mmdet3d/evaluation/functional/waymo_utils/
```

接下来，您就可以在 Waymo 上评估您的模型了。如下示例是使用 8 个图形处理器 (GPU) 在 Waymo 上用 Waymo 评价指标评估 PointPillars 模型的情景：

```shell
./tools/dist_test.sh configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car.py checkpoints/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car_latest.pth
```

如果需要生成 bin 文件，需要在配置文件的 `test_evaluator` 中指定 `pklfile_prefix`，因此你可以在命令后添加 `--cfg-options "test_evaluator.pklfile_prefix=xxxx"`。

**注意**:

1. 有时用 `bazel` 构建 `compute_detection_metrics_main` 的过程中会出现如下错误：`'round' 不是 'std' 的成员` (`'round' is not a member of 'std'`)。我们只需要移除该文件中，`round` 前的 `std::`。

2. 考虑到 Waymo 上评估一次耗时不短，我们建议只在模型训练结束时进行评估。

3. 为了在 CUDA 9 环境使用 TensorFlow，我们建议通过编译 TensorFlow 源码的方式使用。除了官方教程之外，您还可以参考该[链接](https://github.com/SmileTM/Tensorflow2.X-GPU-CUDA9.0)以寻找可能合适的预编译包以及编译源码的实用攻略。

## 测试并提交到官方服务器

如下是一个使用 8 个图形处理器在 Waymo 上测试 PointPillars，生成 bin 文件并提交结果到官方榜单的例子：

如果你想生成 bin 文件并提交到服务器中，在运行测试指令前你需要在配置文件的 `test_evaluator` 中指定 `submission_prefix`。

在生成 bin 文件后，您可以简单地构建二进制文件 `create_submission`，并按照[指示](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md/)创建一个提交文件。下面是一些示例：

```shell
cd ../waymo-od/
bazel build waymo_open_dataset/metrics/tools/create_submission
cp bazel-bin/waymo_open_dataset/metrics/tools/create_submission ../mmdetection3d/mmdet3d/core/evaluation/waymo_utils/
vim waymo_open_dataset/metrics/tools/submission.txtpb  # set the metadata information
cp waymo_open_dataset/metrics/tools/submission.txtpb ../mmdetection3d/mmdet3d/evaluation/functional/waymo_utils/

cd ../mmdetection3d
# suppose the result bin is in `results/waymo-car/submission`
mmdet3d/core/evaluation/waymo_utils/create_submission  --input_filenames='results/waymo-car/kitti_results_test.bin' --output_filename='results/waymo-car/submission/model' --submission_filename='mmdet3d/evaluation/functional/waymo_utils/submission.txtpb'

tar cvf results/waymo-car/submission/my_model.tar results/waymo-car/submission/my_model/
gzip results/waymo-car/submission/my_model.tar
```

如果想用官方评估服务器评估您在验证集上的结果，您可以使用同样的方法生成提交文件，只需确保您在运行如上指令前更改 `submission.txtpb` 中的字段值即可。
