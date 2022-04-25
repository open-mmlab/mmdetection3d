# 基于激光雷达的 3D 语义分割

基于激光雷达的 3D 语义分割是 MMDetection3D 支持的最基础的任务之一。它期望给定的模型以激光雷达采集的任意数量的特征点为输入，并预测每个输入点的语义标签。接下来，我们以 ScanNet 数据集上的 PointNet++ (SSG) 为例，展示如何准备数据，在标准的 3D 语义分割基准上训练并测试模型，以及可视化并验证结果。

## 数据准备

首先，我们需要从 ScanNet [官方网站](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation)下载原始数据。

由于不同数据集的原始数据有不同的组织方式，我们通常需要用 pkl 或 json 文件收集有用的数据信息。

因此，在准备好所有的原始数据之后，我们可以遵循 [ScanNet 文档](https://github.com/open-mmlab/mmdetection3d/blob/master/data/scannet/README.md/)中的说明生成数据信息。

随后，相关的目录结构将如下所示：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── scannet
│   │   ├── scannet_utils.py
│   │   ├── batch_load_scannet_data.py
│   │   ├── load_scannet_data.py
│   │   ├── scannet_utils.py
│   │   ├── README.md
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── scannet_instance_data
│   │   ├── points
│   │   ├── instance_mask
│   │   ├── semantic_mask
│   │   ├── seg_info
│   │   │   ├── train_label_weight.npy
│   │   │   ├── train_resampled_scene_idxs.npy
│   │   │   ├── val_label_weight.npy
│   │   │   ├── val_resampled_scene_idxs.npy
│   │   ├── scannet_infos_train.pkl
│   │   ├── scannet_infos_val.pkl
│   │   ├── scannet_infos_test.pkl
```

## 训练

接着，我们将使用提供的配置文件训练 PointNet++ (SSG) 模型。当你使用不同的 GPU 设置进行训练时，你基本上可以按照这个[教程](https://mmdetection3d.readthedocs.io/zh_CN/latest/1_exist_data_model.html#inference-with-existing-models)的示例脚本。假设我们在一台具有 2 块 GPU 的机器上使用分布式训练：

```
./tools/dist_train.sh configs/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py 2
```

注意，配置文件名中的 `16x2` 是指训练时用了 2 块 GPU，每块 GPU 上有 16 个样本。如果你的自定义设置不同于此，那么有时候你需要相应的调整学习率。基本规则可以参考[此处](https://arxiv.org/abs/1706.02677)。

## 定量评估

在训练期间，模型权重将会根据配置文件中的 `evaluation = dict(interval=xxx)` 设置被周期性地评估。我们支持不同数据集的官方评估方案。对于 ScanNet，将使用 20 个类别的平均交并比 (mIoU) 对模型进行评估。评估结果将会被打印到终端中，如下所示：

```
+---------+--------+--------+---------+--------+--------+--------+--------+--------+--------+-----------+---------+---------+--------+---------+--------------+----------------+--------+--------+---------+----------------+--------+--------+---------+
| classes | wall   | floor  | cabinet | bed    | chair  | sofa   | table  | door   | window | bookshelf | picture | counter | desk   | curtain | refrigerator | showercurtrain | toilet | sink   | bathtub | otherfurniture | miou   | acc    | acc_cls |
+---------+--------+--------+---------+--------+--------+--------+--------+--------+--------+-----------+---------+---------+--------+---------+--------------+----------------+--------+--------+---------+----------------+--------+--------+---------+
| results | 0.7257 | 0.9373 | 0.4625  | 0.6613 | 0.7707 | 0.5562 | 0.5864 | 0.4010 | 0.4558 | 0.7011    | 0.2500  | 0.4645  | 0.4540 | 0.5399  | 0.2802       | 0.3488         | 0.7359 | 0.4971 | 0.6922  | 0.3681         | 0.5444 | 0.8118 | 0.6695  |
+---------+--------+--------+---------+--------+--------+--------+--------+--------+--------+-----------+---------+---------+--------+---------+--------------+----------------+--------+--------+---------+----------------+--------+--------+---------+
```

此外，在训练完成后你也可以评估特定的模型权重文件。你可以简单地执行以下脚本：

```
./tools/dist_test.sh configs/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py \
    work_dirs/pointnet2_ssg/latest.pth --eval mIoU
```

## 测试与提交

如果你只想在在线基准上进行推理或测试模型性能，你需要将之前评估脚本中的 `--eval mIoU` 替换成 `--format-only`，并将 ScanNet 数据集[配置文件](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/scannet_seg-3d-20class.py#L126)中的 `ann_file=data_root + 'scannet_infos_val.pkl'` 变成 `ann_file=data_root + 'scannet_infos_test.pkl'`。记住将 `txt_prefix` 指定为保存测试结果的目录，例如，添加选项 `--eval-options txt_prefix=work_dirs/pointnet2_ssg/test_submission`。在生成结果后，你可以压缩文件夹并上传至 [ScanNet 评估服务器](http://kaldir.vc.in.tum.de/scannet_benchmark/semantic_label_3d)上。

## 定性评估

MMDetection3D 还提供了通用的可视化工具，以便于我们可以对训练好的模型预测的分割结果有一个直观的感受。你也可以在评估阶段通过设置 `--eval-options 'show=True' 'out_dir=${SHOW_DIR}'` 来在线可视化分割结果，或者使用 `tools/misc/visualize_results.py` 来离线地进行可视化。此外，我们还提供了脚本 `tools/misc/browse_dataset.py` 用于可视化数据集而不做推理。更多的细节请参考[可视化文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/useful_tools.html#visualization)。
