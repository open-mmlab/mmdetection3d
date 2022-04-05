# 教程 7: 模型部署

为了解决在实际使用过程中对算法模型的性能需求，通常我们会将训练好的模型部署到各种推理后端上。 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 是 OpenMMLab 系列算法库的部署框架，现在 MMDeploy 已经支持了MMDet3d，我们可以通过 MMDeploy 将训练好的模型部署到各种推理后端上。

## 1.准备

### 安装MMDeploy

```bash
git clone -b master git@github.com:open-mmlab/mmdeploy.git
cd mmdeploy
git submodule update --init --recursive
```

### 安装推理后端编译自定义算子

根据 MMDeploy 的文档选择安装推理后端并编译自定义算子，目前 MMDet3D 模型支持了的推理后端有[OnnxRuntime](https://mmdeploy.readthedocs.io/en/latest/backends/onnxruntime.html)，[Tensorrt](https://mmdeploy.readthedocs.io/en/latest/backends/tensorrt.html)，[OpenVino](https://mmdeploy.readthedocs.io/en/latest/backends/openvino.html)。

## 2.转换模型

通过 MMDeploy 将 MMDet3D 训练好的模型转换成 ONNX 模型和推理后端所需要的模型。

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

* deploy_cfg : MMDeploy 中用于部署的配置文件路径。
* model_cfg : OpenMMLab 系列代码库中使用的模型配置文件路径。
* checkpoint : OpenMMLab 系列代码库的模型文件路径。
* img : 用于模型转换时使用的图像文件路径。
* --test-img : 用于测试模型的图像文件路径。默认设置成None。
* --work-dir : 工作目录，用来保存日志和模型文件。
* --calib-dataset-cfg : 此参数只有int8模式下生效，用于校准
* 数据集配置文件。若在int8模式下未传入参数，则会自动使用模型配置文件中的'val'数据集进行校准。
* --device : 用于模型转换的设备。 默认是cpu。
* --log-level : 设置日记的等级，选项包括'CRITICAL'， 'FATAL'， 'ERROR'， 'WARN'， 'WARNING'， 'INFO'， 'DEBUG'， 'NOTSET'。 默认是INFO。
* --show : 是否显示检测的结果。
* --dump-info : 是否输出 SDK 信息。

### 示例
```bash
cd mmdeploy \
python tools/deploy.py \
configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-kitti.py \
${$MMDET3D_DIR}/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py \
${$MMDET3D_DIR}/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth \
${$MMDET3D_DIR}/demo/data/kitti/kitti_000008.bin \
--work-dir
work-dir \
--device
cuda:0 \
--show
```

## 3.测试模型(可选)

可以在数据集上测试部署在推理后端上的模型的精度和速度。

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
[--metric-options ${METRIC_OPTIONS}]
[--log2file work_dirs/output.txt]
```

### 示例

```bash
cd mmdeploy \
python tools/test.py \
configs/mmdet3d/voxel-detection/voxel-detection_onnxruntime_dynamic.py \
${MMDET3D_DIR}/configs/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py \
--model
work-dir/end2end.onnx \
--metrics
bbox \
--device
cpu
```

## 支持模型列表

| Model                     | Codebase         | TorchScript | OnnxRuntime | TensorRT | NCNN | PPLNN | OpenVINO | Model config                                                                                   |
|---------------------------|------------------|:-----------:|:-----------:|:--------:|:----:|:-----:|:--------:|------------------------------------------------------------------------------------------------|
| PointPillars              | MMDetection3d    |      ?      |      Y      |    Y     |  N   |   N   |    Y     | [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars)         |
| CenterPoint (pillar)      | MMDetection3d    |      ?      |      Y      |    Y     |   N   |   N   |    Y     | [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/centerpoint)          |

## 注意
目前 centerpoint 仅支持了 pillar 版本的。
