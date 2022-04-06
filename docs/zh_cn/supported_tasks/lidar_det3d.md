# 基于 LiDAR 的 3D 检测

基于 LiDAR 的 3D 检测算法是 MMDetection3D 支持的最基础的任务之一。对于给定的算法模型，输入为任意数量的、附有 LiDAR 采集的特征的点，输出为每个感兴趣目标的 3D 矩形框 (Bounding Box) 和类别标签。接下来，我们将以在 KITTI 数据集上训练 PointPillars 为例，介绍如何准备数据，如何在标准 3D 检测基准数据集上训练和测试模型，以及如何可视化并验证结果。

## 数据预处理

最开始，我们需要下载原始数据，并按[文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/data_preparation.html)中介绍的那样，把数据重新整理成标准格式。值得注意的是，对于 KIITI 数据集，我们需要额外的 txt 文件用于数据整理。

由于不同数据集上的原始数据有不同的组织方式，我们通常需要用 .pkl 或者 .json 文件收集有用的数据信息。在准备好原始数据后，我们需要运行脚本 `create_data.py`，为不同的数据集生成数据。如，对于 KITTI 数据集，我们需要执行:

```
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

随后，相对目录结构将变成如下形式：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
│   │   ├── kitti_gt_database
│   │   ├── kitti_infos_train.pkl
│   │   ├── kitti_infos_trainval.pkl
│   │   ├── kitti_infos_val.pkl
│   │   ├── kitti_infos_test.pkl
│   │   ├── kitti_dbinfos_train.pkl
```

## 训练

接着，我们将使用提供的配置文件训练 PointPillars。当你使用不同的 GPU 设置进行训练时，你基本上可以按照这个[教程](https://mmdetection3d.readthedocs.io/zh_CN/latest/1_exist_data_model.html)的示例脚本进行训练。假设我们在一台具有 8 块 GPU 的机器上进行分布式训练：

```
./tools/dist_train.sh configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py 8
```

注意到，配置文件名字中的 `6x8` 是指训练时是用了 8 块 GPU，每块 GPU 上有 6 个样本。如果你有不同的自定义的设置，那么有时你可能需要调整学习率。可以参考这篇[文献](https://arxiv.org/abs/1706.02677)。

## 定量评估

在训练期间，模型将会根据配置文件中的 `evaluation = dict(interval=xxx)` 设置，被周期性地评估。我们支持不同数据集的官方评估方案。对于 KITTI, 模型的评价指标为平均精度 (mAP, mean average precision)。3 种类型的 mAP 的交并比 (IoU, Intersection over Union) 阈值可以取 0.5/0.7。评估结果将会被打印到终端中，如下所示：

```
Car AP@0.70, 0.70, 0.70:
bbox AP:98.1839, 89.7606, 88.7837
bev AP:89.6905, 87.4570, 85.4865
3d AP:87.4561, 76.7569, 74.1302
aos AP:97.70, 88.73, 87.34
Car AP@0.70, 0.50, 0.50:
bbox AP:98.1839, 89.7606, 88.7837
bev AP:98.4400, 90.1218, 89.6270
3d AP:98.3329, 90.0209, 89.4035
aos AP:97.70, 88.73, 87.34
```

评估某个特定的模型权重文件。你可以简单地执行下列的脚本：

```
./tools/dist_test.sh configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py \
    work_dirs/pointpillars/latest.pth --eval mAP
```

## 测试与提交

如果你只想在线上基准上进行推理或者测试模型的表现，你只需要把上面评估脚本中的 `--eval mAP` 替换为 `--format-only`。如果需要的话，还可以指定 `pklfile_prefix` 和 `submission_prefix`，如，添加命令行选项 `--eval-options submission_prefix=work_dirs/pointpillars/test_submission`。请确保配置文件中的[测试信息](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/kitti-3d-3class.py#L131)与测试集对应，而不是验证集。在生成结果后，你可以压缩文件夹，并上传到 KITTI 的评估服务器上。

## 定性验证

MMDetection3D 还提供了通用的可视化工具，以便于我们可以对训练好的模型的预测结果有一个直观的感受。你可以在命令行中添加 `--eval-options 'show=True' 'out_dir=${SHOW_DIR}'` 选项，在评估过程中在线地可视化检测结果；你也可以使用 `tools/misc/visualize_results.py`, 离线地进行可视化。另外，我们还提供了脚本 `tools/misc/browse_dataset.py`， 可视化数据集而不做推理。更多的细节请参考[可视化文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/useful_tools.html#id2)
