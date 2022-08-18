# Tutorial 8: MMDetection3D model deployment

To meet the speed requirement of the model in practical use, usually, we deploy the trained model to inference backends. [MMDeploy](https://github.com/open-mmlab/mmdeploy) is OpenMMLab model deployment framework. Now MMDeploy has supported MMDetection3D model deployment, and you can deploy the trained model to inference backends by MMDeploy.

## Prerequisite

### Install MMDeploy

```bash
git clone -b master git@github.com:open-mmlab/mmdeploy.git
cd mmdeploy
git submodule update --init --recursive
```

### Install backend and build custom ops

According to MMDeploy documentation, choose to install the inference backend and build custom ops. Now supported inference backends for MMDetection3D include [OnnxRuntime](https://mmdeploy.readthedocs.io/en/latest/backends/onnxruntime.html), [TensorRT](https://mmdeploy.readthedocs.io/en/latest/backends/tensorrt.html), [OpenVINO](https://mmdeploy.readthedocs.io/en/latest/backends/openvino.html).

## Export model

Export the Pytorch model of MMDetection3D to the ONNX model file and the model file required by the backend. You could refer to MMDeploy docs [how to convert model](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_convert_model.html).

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

### Description of all arguments

- `deploy_cfg` : The path of deploy config file in MMDeploy codebase.
- `model_cfg` : The path of model config file in OpenMMLab codebase.
- `checkpoint` : The path of model checkpoint file.
- `img` : The path of point cloud file or image file that used to convert model.
- `--test-img` : The path of image file that used to test model. If not specified, it will be set to `None`.
- `--work-dir` : The path of work directory that used to save logs and models.
- `--calib-dataset-cfg` : Only valid in int8 mode. Config used for calibration. If not specified, it will be set to `None` and  use "val" dataset in model config for calibration.
- `--device` : The device used for conversion. If not specified, it will be set to `cpu`.
- `--log-level` : To set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`.
- `--show` : Whether to show detection outputs.
- `--dump-info` : Whether to output information for SDK.

### Example

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

## Inference Model

Now you can do model inference with the APIs provided by the backend. But what if you want to test the model instantly? We have some backend wrappers for you.

```python
from mmdeploy.apis import inference_model

result = inference_model(model_cfg, deploy_cfg, backend_files, img=img, device=device)
```

The `inference_model` will create a wrapper module and do the inference for you. The result has the same format as the original OpenMMLab repo.

## Evaluate model (Optional)

You can test the accuracy and speed of the model in the inference backend. You could refer to MMDeploy docs [how to measure performance of models](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_measure_performance_of_models.html).

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

### Example

```bash
cd mmdeploy
python tools/test.py \
    configs/mmdet3d/voxel-detection/voxel-detection_onnxruntime_dynamic.py \
    ${MMDET3D_DIR}/configs/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py \
    --model work-dir/end2end.onnx \
    --metrics bbox \
    --device cpu
```

## Supported models

| Model                | TorchScript | OnnxRuntime | TensorRT | NCNN | PPLNN | OpenVINO | Model config                                                                           |
| -------------------- | :---------: | :---------: | :------: | :--: | :---: | :------: | -------------------------------------------------------------------------------------- |
| PointPillars         |      ?      |      Y      |    Y     |  N   |   N   |    Y     | [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars) |
| CenterPoint (pillar) |      ?      |      Y      |    Y     |  N   |   N   |    Y     | [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/centerpoint)  |

## Note

- MMDeploy version >= 0.4.0.
- Currently, CenterPoint has only supported the pillar version.
