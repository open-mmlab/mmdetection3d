# Useful Tools and Scripts

We provide lots of useful tools under `tools/` directory.

## Log Analysis

You can plot loss/mAP curves given a training log file. Run `pip install seaborn` first to install the dependency.

![loss curve image](../resources/loss_curve.png)

```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

- Plot the classification loss of some run.

  ```shell
  python tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
  ```

- Plot the classification and regression loss of some run, and save the figure to a pdf.

  ```shell
  python tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
  ```

- Compare the bbox mAP of two runs in the same figure.

  ```shell
  python tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
  ```

You can also compute the average training speed.

```shell
python tools/analyze_logs.py cal_train_time log.json [--include-outliers]
```

The output is expected to be like the following.

```
-----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
slowest epoch 11, average time is 1.2024
fastest epoch 1, average time is 1.1909
time std over epochs is 0.0028
average iter time: 1.1959 s/iter
```

## Visualization

To see the SUNRGBD, ScanNet or KITTI points and detection results, you can run the following command

```bash
python tools/test.py ${CONFIG_FILE} ${CKPT_PATH} --show --show-dir ${SHOW_DIR}
```

Aftering running this command, plotted results ***_points.obj and ***_pred.ply files in `${SHOW_DIR}`.

To see the points, detection results and ground truth of SUNRGBD, ScanNet or KITTI during evaluation time, you can run the following command
```bash
python tools/test.py ${CONFIG_FILE} ${CKPT_PATH} --eval 'mAP' --options 'show=True' 'out_dir=${SHOW_DIR}'
```
After running this command, you will obtain ***_points.ob, ***_pred.ply files and ***_gt.ply in `${SHOW_DIR}`.

You can use 3D visualization software such as the [MeshLab](http://www.meshlab.net/) to open the these files under `${SHOW_DIR}` to see the 3D detection output. Specifically, open `***_points.obj` to see the input point cloud and open `***_pred.ply` to see the predicted 3D bounding boxes. This allows the inference and results generation be done in remote server and the users can open them on their host with GUI.

**Notice**: The visualization API is a little unstable since we plan to refactor these parts together with MMDetection in the future.

## Model Complexity

`tools/get_flops.py` is a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

You will get the results like this.

```text
==============================
Input shape: (3, 1280, 800)
Flops: 239.32 GFLOPs
Params: 37.74 M
==============================
```

**Note**: This tool is still experimental and we do not guarantee that the
 number is absolutely correct. You may well use the result for simple
  comparisons, but double check it before you adopt it in technical reports or papers.

1. FLOPs are related to the input shape while parameters are not. The default
 input shape is (1, 3, 1280, 800).
2. Some operators are not counted into FLOPs like GN and custom operators. Refer to [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) for details.
3. The FLOPs of two-stage detectors is dependent on the number of proposals.

## Model Conversion

### RegNet model to MMDetection

`tools/regnet2mmdet.py` convert keys in pycls pretrained RegNet models to
 MMDetection style.

```shell
python tools/regnet2mmdet.py ${SRC} ${DST} [-h]
```

### Detectron ResNet to Pytorch

`tools/detectron2pytorch.py` converts keys in the original detectron pretrained
 ResNet models to PyTorch style.

```shell
python tools/detectron2pytorch.py ${SRC} ${DST} ${DEPTH} [-h]
```

### Prepare a model for publishing

`tools/publish_model.py` helps users to prepare their model for publishing.

Before you upload a model to AWS, you may want to

1. convert model weights to CPU tensors
2. delete the optimizer states and
3. compute the hash of the checkpoint file and append the hash id to the
 filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/faster_rcnn/latest.pth faster_rcnn_r50_fpn_1x_20190801.pth
```

The final output filename will be `faster_rcnn_r50_fpn_1x_20190801-{hash id}.pth`.

## Dataset Conversion

TBD

## Miscellaneous

### Print the entire config

`tools/print_config.py` prints the whole config verbatim, expanding all its
 imports.

```shell
python tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```
