# Waymo 数据集

本文档页包含了关于 MMDetection3D 中 Waymo 数据集用法的教程。

## 数据集准备

在准备 Waymo 数据集之前，如果您之前只安装了 `requirements/build.txt` 和 `requirements/runtime.txt` 中的依赖，请通过运行如下指令额外安装 Waymo 数据集所依赖的官方包：

```
# tf 2.1.0.
pip install waymo-open-dataset-tf-2-1-0==1.2.0
# tf 2.0.0
# pip install waymo-open-dataset-tf-2-0-0==1.2.0
# tf 1.15.0
# pip install waymo-open-dataset-tf-1-15-0==1.2.0
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
│   │   ├── kitti_format
│   │   │   ├── ImageSets

```

您可以在[这里](https://waymo.com/open/download/)下载 1.2 版本的 Waymo 公开数据集，并在[这里](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o?usp=sharing)下载其训练/验证/测试集拆分文件。接下来，请将 `tfrecord` 文件放入 `data/waymo/waymo_format/` 下的对应文件夹，并将 txt 格式的数据集拆分文件放入 `data/waymo/kitti_format/ImageSets`。在[这里](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0/validation/ground_truth_objects)下载验证集使用的 bin 格式真实标注 (Ground Truth) 文件并放入 `data/waymo/waymo_format/`。小窍门：您可以使用 `gsutil` 来在命令行下载大规模数据集。您可以将该[工具](https://github.com/RalphMao/Waymo-Dataset-Tool) 作为一个例子来查看更多细节。之后，通过运行如下指令准备 Waymo 数据：

```bash
python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --workers 128 --extra-tag waymo
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
│   │   ├── kitti_format
│   │   │   ├── ImageSets
│   │   │   ├── training
│   │   │   │   ├── calib
│   │   │   │   ├── image_0
│   │   │   │   ├── image_1
│   │   │   │   ├── image_2
│   │   │   │   ├── image_3
│   │   │   │   ├── image_4
│   │   │   │   ├── label_0
│   │   │   │   ├── label_1
│   │   │   │   ├── label_2
│   │   │   │   ├── label_3
│   │   │   │   ├── label_4
│   │   │   │   ├── label_all
│   │   │   │   ├── pose
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

因为 Waymo 数据的来源包含数个相机，这里我们将每个相机对应的图像和标签文件分别存储，并将相机位姿 (pose) 文件存储下来以供后续处理连续多帧的点云。我们使用 `{a}{bbb}{ccc}` 的名称编码方式为每帧数据命名，其中 `a` 是不同数据拆分的前缀（`0` 指代训练集，`1` 指代验证集，`2` 指代测试集），`bbb` 是分割部分 (segment) 的索引，而 `ccc` 是帧索引。您可以轻而易举地按照如上命名规则定位到所需的帧。我们将训练和验证所需数据按 KITTI 的方式集合在一起，然后将训练集/验证集/测试集的索引存储在 `ImageSet` 下的文件中。

## 训练

考虑到原始数据集中的数据有很多相似的帧，我们基本上可以主要使用一个子集来训练我们的模型。在我们初步的基线中，我们在每五帧图片中加载一帧。得益于我们的超参数设置和数据增强方案，我们得到了比 Waymo [原论文](https://arxiv.org/pdf/1912.04838.pdf)中更好的性能。请移步 `configs/pointpillars/` 下的 README.md 以查看更多配置和性能相关的细节。我们会尽快发布一个更完整的 Waymo 基准榜单 (benchmark)。

## 评估

为了在 Waymo 数据集上进行检测性能评估，请按照[此处指示](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md/)构建用于计算评估指标的二进制文件 `compute_detection_metrics_main`，并将它置于 `mmdet3d/core/evaluation/waymo_utils/` 下。您基本上可以按照下方命令安装 `bazel`，然后构建二进制文件：

   ```shell
   git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
   cd waymo-od
   git checkout remotes/origin/master

   sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
   wget https://github.com/bazelbuild/bazel/releases/download/0.28.0/bazel-0.28.0-installer-linux-x86_64.sh
   sudo bash bazel-0.28.0-installer-linux-x86_64.sh
   sudo apt install build-essential

   ./configure.sh
   bazel clean

   bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
   cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main ../mmdetection3d/mmdet3d/core/evaluation/waymo_utils/
   ```

接下来，您就可以在 Waymo 上评估您的模型了。如下示例是使用 8 个图形处理器 (GPU) 在 Waymo 上用 Waymo 评价指标评估 PointPillars 模型的情景：

   ```shell
   ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} configs/pointpillars/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car.py \
       checkpoints/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car_latest.pth --out results/waymo-car/results_eval.pkl \
       --eval waymo --eval-options 'pklfile_prefix=results/waymo-car/kitti_results' \
       'submission_prefix=results/waymo-car/kitti_results'
   ```

如果需要生成 bin 文件，应在 `--eval-options` 中给出 `pklfile_prefix`。对于评价指标， `waymo` 是我们推荐的官方评估原型。目前，`kitti` 这一评估选项是从 KITTI 迁移而来的，且每个难度下的评估结果和 KITTI 数据集中定义得到的不尽相同——目前大多数物体被标记为难度 0（日后会修复）。`kitti` 评估选项的不稳定来源于很大的计算量，转换的数据中遮挡 (occlusion) 和截断 (truncation) 的缺失，难度的不同定义方式，以及不同的平均精度 (Average Precision) 计算方式。

**注意**:

1. 有时用 `bazel` 构建 `compute_detection_metrics_main` 的过程中会出现如下错误：`'round' 不是 'std' 的成员` (`'round' is not a member of 'std'`)。我们只需要移除该文件中，`round` 前的 `std::`。

2. 考虑到 Waymo 上评估一次耗时不短，我们建议只在模型训练结束时进行评估。

3. 为了在 CUDA 9 环境使用 TensorFlow，我们建议通过编译 TensorFlow 源码的方式使用。除了官方教程之外，您还可以参考该[链接](https://github.com/SmileTM/Tensorflow2.X-GPU-CUDA9.0)以寻找可能合适的预编译包以及编译源码的实用攻略。

## 测试并提交到官方服务器

如下是一个使用 8 个图形处理器在 Waymo 上测试 PointPillars，生成 bin 文件并提交结果到官方榜单的例子：

   ```shell
   ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} configs/pointpillars/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car.py \
       checkpoints/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car_latest.pth --out results/waymo-car/results_eval.pkl \
       --format-only --eval-options 'pklfile_prefix=results/waymo-car/kitti_results' \
       'submission_prefix=results/waymo-car/kitti_results'
   ```

在生成 bin 文件后，您可以简单地构建二进制文件 `create_submission`，并按照[指示](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md/) 创建一个提交文件。下面是一些示例：

   ```shell
   cd ../waymo-od/
   bazel build waymo_open_dataset/metrics/tools/create_submission
   cp bazel-bin/waymo_open_dataset/metrics/tools/create_submission ../mmdetection3d/mmdet3d/core/evaluation/waymo_utils/
   vim waymo_open_dataset/metrics/tools/submission.txtpb  # set the metadata information
   cp waymo_open_dataset/metrics/tools/submission.txtpb ../mmdetection3d/mmdet3d/core/evaluation/waymo_utils/

   cd ../mmdetection3d
   # suppose the result bin is in `results/waymo-car/submission`
   mmdet3d/core/evaluation/waymo_utils/create_submission  --input_filenames='results/waymo-car/kitti_results_test.bin' --output_filename='results/waymo-car/submission/model' --submission_filename='mmdet3d/core/evaluation/waymo_utils/submission.txtpb'

   tar cvf results/waymo-car/submission/my_model.tar results/waymo-car/submission/my_model/
   gzip results/waymo-car/submission/my_model.tar
   ```

如果想用官方评估服务器评估您在验证集上的结果，您可以使用同样的方法生成提交文件，只需确保您在运行如上指令前更改 `submission.txtpb` 中的字段值即可。
