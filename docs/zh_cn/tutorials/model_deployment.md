# 教程 8: MMDet3D 模型部署

为了满足在实际使用过程中遇到的算法模型的速度需求，通常我们会将训练好的模型部署到各种推理后端上。 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 是 OpenMMLab 系列算法库的部署框架，现在 MMDeploy 已经支持了 MMDetection3D，我们可以通过 MMDeploy 将训练好的模型部署到各种推理后端上。

## 准备

### 安装 MMDeploy

```bash
git clone -b master git@github.com:open-mmlab/mmdeploy.git
cd mmdeploy
git submodule update --init --recursive
```

### 安装推理后端编译自定义算子

根据 MMDeploy 的文档选择安装推理后端并编译自定义算子，目前 MMDet3D 模型支持了的推理后端有 [OnnxRuntime](https://mmdeploy.readthedocs.io/en/latest/backends/onnxruntime.html)，[TensorRT](https://mmdeploy.readthedocs.io/en/latest/backends/tensorrt.html)，[OpenVINO](https://mmdeploy.readthedocs.io/en/latest/backends/openvino.html)。

## 模型导出

将 MMDet3D 训练好的 Pytorch 模型转换成 ONNX 模型文件和推理后端所需要的模型文件。你可以参考 MMDeploy 的文档 [how_to_convert_model.md](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorials/how_to_convert_model.md)。

```bash
python ./tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    ${INPUT_IMG} \
    --test-img ${TEST_IMG} \
    --work-dir ${WORK_DIR} \
    --calib-dataset-cfg ${CALIB_DATA_CFG} \
    --device ${DEVICE} \
    --log-level INFO \
    --show \
    --dump-info
```

### 参数描述

- `deploy_cfg` : MMDeploy 代码库中用于部署的配置文件路径。
- `model_cfg` : OpenMMLab 系列代码库中使用的模型配置文件路径。
- `checkpoint` : OpenMMLab 系列代码库的模型文件路径。
- `img` : 用于模型转换时使用的点云文件或图像文件路径。
- `--test-img` : 用于测试模型的图像文件路径。如果没有指定，将设置成 `None`。
- `--work-dir` : 工作目录，用来保存日志和模型文件。
- `--calib-dataset-cfg` : 此参数只在 int8 模式下生效，用于校准数据集配置文件。如果没有指定，将被设置成 `None`，并使用模型配置文件中的 'val' 数据集进行校准。
- `--device` : 用于模型转换的设备。如果没有指定，将被设置成 cpu。
- `--log-level` : 设置日记的等级，选项包括 `'CRITICAL'，'FATAL'，'ERROR'，'WARN'，'WARNING'，'INFO'，'DEBUG'，'NOTSET'`。如果没有指定，将被设置成 INFO。
- `--show` : 是否显示检测的结果。
- `--dump-info` : 是否输出 SDK 信息。

### 示例

```bash
cd mmdeploy
python tools/deploy.py \
    configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-kitti.py \
    ${$MMDET3D_DIR}/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py \
    ${$MMDET3D_DIR}/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth \
    ${$MMDET3D_DIR}/demo/data/kitti/kitti_000008.bin \
    --work-dir work-dir \
    --device cuda:0 \
    --show
```

## 模型推理

现在你可以使用推理后端提供的 API 进行模型推理。但是，如果你想立即测试模型怎么办？我们为您准备了一些推理后端的封装。

```python
from mmdeploy.apis import inference_model

result = inference_model(model_cfg, deploy_cfg, backend_files, img=img, device=device)
```

`inference_model` 将创建一个推理后端的模块并为你进行推理。推理结果与模型的 OpenMMLab 代码库具有相同的格式。

## 测试模型（可选）

可以测试部署在推理后端上的模型的精度和速度。你可以参考 [how to measure performance of models](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_measure_performance_of_models.html)。

```bash
python tools/test.py \
    ${DEPLOY_CFG} \
    ${MODEL_CFG} \
    --model ${BACKEND_MODEL_FILES} \
    [--out ${OUTPUT_PKL_FILE}] \
    [--format-only] \
    [--metrics ${METRICS}] \
    [--show] \
    [--show-dir ${OUTPUT_IMAGE_DIR}] \
    [--show-score-thr ${SHOW_SCORE_THR}] \
    --device ${DEVICE} \
    [--cfg-options ${CFG_OPTIONS}] \
    [--metric-options ${METRIC_OPTIONS}] \
    [--log2file work_dirs/output.txt]
```

### 示例

```bash
cd mmdeploy
python tools/test.py \
    configs/mmdet3d/voxel-detection/voxel-detection_onnxruntime_dynamic.py \
    ${MMDET3D_DIR}/configs/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py \
    --model work-dir/end2end.onnx \
    --metrics bbox \
    --device cpu
```

## 支持模型列表

| Model                | TorchScript | OnnxRuntime | TensorRT | NCNN | PPLNN | OpenVINO | Model config                                                                           |
| -------------------- | :---------: | :---------: | :------: | :--: | :---: | :------: | -------------------------------------------------------------------------------------- |
| PointPillars         |      ?      |      Y      |    Y     |  N   |   N   |    Y     | [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars) |
| CenterPoint (pillar) |      ?      |      Y      |    Y     |  N   |   N   |    Y     | [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/centerpoint)  |

## 注意

- MMDeploy 的版本需要 >= 0.4.0。
- 目前 CenterPoint 仅支持了 pillar 版本的。
