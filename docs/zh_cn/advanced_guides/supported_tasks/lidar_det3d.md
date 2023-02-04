# 基于激光雷达的 3D 检测

基于激光雷达的 3D 检测是 MMDetection3D 支持的最基础的任务之一。它期望给定的模型以激光雷达采集的任意数量的特征点为输入，并为每一个感兴趣的目标预测 3D 框及类别标签。接下来，我们以 KITTI 数据集上的 PointPillars 为例，展示如何准备数据，在标准的 3D 检测基准上训练并测试模型，以及可视化并验证结果。

## 数据准备

首先，我们需要下载原始数据并按照[数据准备文档](https://mmdetection3d.readthedocs.io/zh_CN/dev-1.x/user_guides/dataset_prepare.html)中提供的标准方式重新组织数据。

由于不同数据集的原始数据有不同的组织方式，我们通常需要用 `.pkl` 文件收集有用的数据信息。因此，在准备好所有的原始数据之后，我们需要运行 `create_data.py` 中提供的脚本来为不同的数据集生成数据集信息。例如，对于 KITTI，我们需要运行如下命令：

```shell
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

随后，相关的目录结构将如下所示：

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
│   │   │   ├── velodyne_reduced
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
│   │   │   ├── velodyne_reduced
│   │   ├── kitti_gt_database
│   │   ├── kitti_infos_train.pkl
│   │   ├── kitti_infos_trainval.pkl
│   │   ├── kitti_infos_val.pkl
│   │   ├── kitti_infos_test.pkl
│   │   ├── kitti_dbinfos_train.pkl
```

## 训练

接着，我们将使用提供的配置文件训练 PointPillars。当您使用不同的 GPU 设置进行训练时，您可以按照这个[教程](https://mmdetection3d.readthedocs.io/en/dev-1.x/user_guides/train_test.html)的示例。假设我们在一台具有 8 块 GPU 的机器上使用分布式训练：

```shell
./tools/dist_train.sh configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py 8
```

注意，配置文件名中的 `8xb6` 是指训练用了 8 块 GPU，每块 GPU 上有 6 个数据样本。如果您的自定义设置不同于此，那么有时候您需要相应地调整学习率。基本规则可以参考[此处](https://arxiv.org/abs/1706.02677)。我们已经支持了使用 `--auto-scale-lr` 来自动缩放学习率。

## 定量评估

在训练期间，模型权重文件将会根据配置文件中的 `train_cfg = dict(val_interval=xxx)` 设置被周期性地评估。我们支持不同数据集的官方评估方案。对于 KITTI，将对 3 个类别使用交并比（IoU）阈值分别为 0.5/0.7 的平均精度（mAP）来评估模型。评估结果将会被打印到终端中，如下所示：

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

此外，在训练完成后您也可以评估特定的模型权重文件。您可以简单地执行以下脚本：

```shell
./tools/dist_test.sh configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py work_dirs/pointpillars/latest.pth 8
```

## 测试与提交

如果您只想在在线基准上进行推理或测试模型性能，您需要在相应的评估器中指定 `submission_prefix`，例如，在配置文件中添加 `test_evaluator = dict(type='KittiMetric', ann_file=data_root + 'kitti_infos_test.pkl', format_only=True, pklfile_prefix='results/kitti-3class/kitti_results', submission_prefix='results/kitti-3class/kitti_results')`，然后可以得到结果文件。请确保配置文件中的[测试信息](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/datasets/kitti-3d-3class.py#L117)的 `data_prefix` 和 `ann_file` 由验证集相应地改为测试集。在生成结果后，您可以压缩文件夹并上传至 KITTI 评估服务器上。

## 定性评估

MMDetection3D 还提供了通用的可视化工具，以便于我们可以对训练好的模型预测的检测结果有一个直观的感受。您也可以在评估阶段通过设置 `--show` 来在线可视化检测结果，或者使用 `tools/misc/visualize_results.py` 来离线地进行可视化。此外，我们还提供了脚本 `tools/misc/browse_dataset.py` 用于可视化数据集而不做推理。更多的细节请参考[可视化文档](https://mmdetection3d.readthedocs.io/zh_CN/dev-1.x/user_guides/visualization.html)。
