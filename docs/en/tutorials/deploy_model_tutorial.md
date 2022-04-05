# Tutorial 7: Deploy model in backend

To solve the performance requirements of the model in actual use, we usually deploy the trained model to inference backends. [MMDeploy](https://github.com/open-mmlab/mmdeploy) is OpenMMLab model deployment framework, Now that MMDeploy supports MMDet3d, we can deploy the trained model to inference backends through MMDeploy.

## 1.Prerequist

### Install MMDeploy

```bash
git clone -b master git@github.com:open-mmlab/mmdeploy.git
cd mmdeploy
git submodule update --init --recursive
```

### Install backend and build custom ops

According to MMDeploy documentation, choose to install the inference backend and build custom ops. Currently, the inference backends supported by the MMDet3D model include [OnnxRuntime](https://mmdeploy.readthedocs.io/en/latest/backends/onnxruntime.html)，[Tensorrt](https://mmdeploy.readthedocs.io/en/latest/backends/tensorrt.html)，[OpenVino](https://mmdeploy.readthedocs.io/en/latest/backends/openvino.html)。

## 2.Convert model

Convert the MMDet3D trained model into the ONNX model and the model required by the inference backend through MMDeploy.

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

* `deploy_cfg` : The path of deploy config file in MMDeploy codebase.
* `model_cfg` : The path of model config file in OpenMMLab codebase.
* `checkpoint` : The path of model checkpoint file.
* `img` : The path of image file that used to convert model.
* `--test-img` : The path of image file that used to test model. If not specified, it will be set to `None`.
* `--work-dir` : The path of work directory that used to save logs and models.
* `--calib-dataset-cfg` : Only valid in int8 mode. Config used for calibration. If not specified, it will be set to `None` and  use "val" dataset in model config for calibration.
* `--device` : The device used for conversion. If not specified, it will be set to `cpu`.
* `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.
* `--show` : Whether to show detection outputs.
* `--dump-info` : Whether to output information for SDK.

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
