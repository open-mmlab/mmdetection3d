# 基于视觉的 3D 检测

基于视觉的 3D 检测是指基于纯视觉输入的 3D 检测方法，例如基于单目、双目和多视图图像的 3D 检测。目前，我们只支持单目和多视图的 3D 检测方法。其他方法也应该与我们的框架兼容，并在将来得到支持。

它期望给定的模型以任意数量的图像作为输入，并为每一个感兴趣的目标预测 3D 框及类别标签。以 nuScenes 数据集 FCOS3D 为例，我们将展示如何准备数据，在标准的 3D 检测基准上训练并测试模型，以及可视化并验证结果。

## 数据准备

首先，我们需要下载原始数据并按照[数据准备文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/data_preparation.html)中提供的标准方式重新组织数据。

由于不同数据集的原始数据有不同的组织方式，我们通常需要用 pkl 或 json 文件收集有用的数据信息。因此，在准备好所有的原始数据之后，我们需要运行 `create_data.py` 中提供的脚本来为不同的数据集生成数据信息。例如，对于 nuScenes，我们需要运行如下命令：

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

随后，相关的目录结构将如下所示：

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_trainval.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
│   │   ├── nuscenes_infos_train_mono3d.coco.json
│   │   ├── nuscenes_infos_trainval_mono3d.coco.json
│   │   ├── nuscenes_infos_val_mono3d.coco.json
│   │   ├── nuscenes_infos_test_mono3d.coco.json
```

注意，此处的 pkl 文件主要用于使用 LiDAR 数据的方法，json 文件用于 2D 检测/纯视觉的 3D 检测。在 v0.13.0 支持单目 3D 检测之前，json 文件只包含 2D 检测的信息，因此如果你需要最新的信息，请切换到 v0.13.0 之后的分支。

## 训练

接着，我们将使用提供的配置文件训练 FCOS3D。基本的脚本与其他模型一样。当你使用不同的 GPU 设置进行训练时，你基本上可以按照这个[教程](https://mmdetection3d.readthedocs.io/zh_CN/latest/1_exist_data_model.html#inference-with-existing-models)的示例。假设我们在一台具有 8 块 GPU 的机器上使用分布式训练：

```
./tools/dist_train.sh configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py 8
```

注意，配置文件名中的 `2x8` 是指训练时用了 8 块 GPU，每块 GPU 上有 2 个数据样本。如果你的自定义设置不同于此，那么有时候你需要相应的调整学习率。基本规则可以参考[此处](https://arxiv.org/abs/1706.02677)。

我们也可以通过运行以下命令微调 FCOS3D，从而达到更好的性能：

```
./tools/dist_train.sh fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py 8
```

通过先前的脚本训练好一个基准模型后，请记得相应的修改[此处](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py#L8)的路径。

## 定量评估

在训练期间，模型权重文件将会根据配置文件中的 `evaluation = dict(interval=xxx)` 设置被周期性地评估。

我们支持不同数据集的官方评估方案。由于输出格式与基于其他模态的 3D 检测相同，因此评估方法也是一样的。

对于 nuScenes，将使用基于距离的平均精度（mAP）以及 nuScenes 检测分数（NDS）分别对 10 个类别进行评估。评估结果将会被打印到终端中，如下所示：

```
mAP: 0.3197
mATE: 0.7595
mASE: 0.2700
mAOE: 0.4918
mAVE: 1.3307
mAAE: 0.1724
NDS: 0.3905
Eval time: 170.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.503   0.577   0.152   0.111   2.096   0.136
truck   0.223   0.857   0.224   0.220   1.389   0.179
bus     0.294   0.855   0.204   0.190   2.689   0.283
trailer 0.081   1.094   0.243   0.553   0.742   0.167
construction_vehicle    0.058   1.017   0.450   1.019   0.137   0.341
pedestrian      0.392   0.687   0.284   0.694   0.876   0.158
motorcycle      0.317   0.737   0.265   0.580   2.033   0.104
bicycle 0.308   0.704   0.299   0.892   0.683   0.010
traffic_cone    0.555   0.486   0.309   nan     nan     nan
barrier 0.466   0.581   0.269   0.169   nan     nan
```

此外，在训练完成后你也可以评估特定的模型权重文件。你可以简单地执行以下脚本：

```
./tools/dist_test.sh configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py \
    work_dirs/fcos3d/latest.pth --eval mAP
```

## 测试与提交

如果你只想在在线基准上进行推理或测试模型性能，你需要将之前评估脚本中的 `--eval mAP` 替换成 `--format-only`，并在需要的情况下指定 `jsonfile_prefix`，例如，添加选项 `--eval-options jsonfile_prefix=work_dirs/fcos3d/test_submission`。请确保配置文件中的[测试信息](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/nus-mono3d.py#L93)由验证集相应地改为测试集。

在生成结果后，你可以压缩文件夹并上传至 nuScenes 3D 检测挑战的 evalAI 评估服务器上。

## 定性评估

MMDetection3D 还提供了通用的可视化工具，以便于我们可以对训练好的模型预测的检测结果有一个直观的感受。你也可以在评估阶段通过设置 `--eval-options 'show=True' 'out_dir=${SHOW_DIR}'` 来在线可视化检测结果，或者使用 `tools/misc/visualize_results.py` 来离线地进行可视化。

此外，我们还提供了脚本 `tools/misc/browse_dataset.py` 用于可视化数据集而不做推理。更多的细节请参考[可视化文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/useful_tools.html#visualization)。

注意，目前我们仅支持纯视觉方法在图像上的可视化。将来我们将集成在前景图以及鸟瞰图（BEV）中的可视化。
