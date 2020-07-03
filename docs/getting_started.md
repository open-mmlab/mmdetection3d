# Getting Started

This page provides basic tutorials about the usage of MMDetection.
For installation instructions, please see [install.md](install.md).

## Prepare datasets

It is recommended to symlink the dataset root to `$MMDETECTION3D/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

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
│   ├── lyft
│   │   ├── v1.01-train
│   │   │   ├── v1.01-train (train_data)
│   │   │   ├── lidar (train_lidar)
│   │   │   ├── images (train_images)
│   │   │   ├── maps (train_maps)
│   │   ├── v1.01-test
│   │   │   ├── v1.01-test (test_data)
│   │   │   ├── lidar (test_lidar)
│   │   │   ├── images (test_images)
│   │   │   ├── maps (test_maps)
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   ├── sample_submission.csv
│   ├── scannet
│   │   ├── meta_data
│   │   ├── scans
│   │   ├── batch_load_scannet_data.py
│   │   ├── load_scannet_data.py
│   │   ├── scannet_utils.py
│   │   ├── README.md
│   ├── sunrgbd
│   │   ├── OFFICIAL_SUNRGBD
│   │   ├── matlab
│   │   ├── sunrgbd_data.py
│   │   ├── sunrgbd_utils.py
│   │   ├── README.md

```

Download nuScenes V1.0 full dataset data [HERE]( https://www.nuscenes.org/download). Prepare nuscenes data by running
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

Download KITTI 3D detection data [HERE](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Prepare kitti data by running
```bash
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

Download Lyft 3D detection data [HERE](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data). Prepare Lyft data by running
```bash
python tools/create_data.py lyft --root-path ./data/lyft --out-dir ./data/lyft --extra-tag lyft --version v1.01
```
Note that we follow the original folder names for clear organization. Please rename the raw folders as shown above.

To prepare scannet data, please see [scannet](../data/scannet/README.md).

To prepare sunrgbd data, please see [sunrgbd](../data/sunrgbd/README.md).


For using custom datasets, please refer to [Tutorials 2: Adding New Dataset](tutorials/new_dataset.md).

## Inference with pretrained models

We provide testing scripts to evaluate a whole dataset (COCO, PASCAL VOC, Cityscapes, etc.),
and also some high-level apis for easier integration to other projects.

### Test a dataset

- single GPU
- single node multiple GPU
- multiple node

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `proposal_fast`, `proposal`, `bbox`, `segm` are available for COCO, `mAP`, `recall` for PASCAL VOC. Cityscapes could be evaluated by `cityscapes` as well as all COCO metrics.
- `--show`: If specified, detection results will be plotted in the silient mode. It is only applicable to single GPU testing and used for debugging and visualization. This should be used with `--show-dir`.
- `--show-dir`: If specified, detection results will be plotted on the `***_points.obj` and `***_pred.ply` files in the specified directory. It is only applicable to single GPU testing and used for debugging and visualization. You do NOT need a GUI available in your environment for using this option.


Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test Faster R-CNN and visualize the results. Press any key for the next image.

```shell
python tools/test.py configs/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
    --show
```

2. Test Faster R-CNN and save the painted images for latter visualization.

```shell
python tools/test.py configs/faster_rcnn_r50_fpn_1x.py \
    checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
    --show-dir faster_rcnn_r50_fpn_1x_results
```

3. Test Faster R-CNN on PASCAL VOC (without saving the test results) and evaluate the mAP.

```shell
python tools/test.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc.py \
    checkpoints/SOME_CHECKPOINT.pth \
    --eval mAP
```

4. Test Mask R-CNN with 8 GPUs, and evaluate the bbox and mask AP.

```shell
./tools/dist_test.sh configs/mask_rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth \
    8 --out results.pkl --eval bbox segm
```

5. Test Mask R-CNN with 8 GPUs, and evaluate the **classwise** bbox and mask AP.

```shell
./tools/dist_test.sh configs/mask_rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth \
    8 --out results.pkl --eval bbox segm --options "classwise=True"
```

6. Test Mask R-CNN on COCO test-dev with 8 GPUs, and generate the json file to be submit to the official evaluation server.

```shell
./tools/dist_test.sh configs/mask_rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth \
    8 --format-only --options "jsonfile_prefix=./mask_rcnn_test-dev_results"
```

You will get two json files `mask_rcnn_test-dev_results.bbox.json` and `mask_rcnn_test-dev_results.segm.json`.

7. Test Mask R-CNN on Cityscapes test with 8 GPUs, and generate the txt and png files to be submit to the official evaluation server.

```shell
./tools/dist_test.sh configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py \
    checkpoints/mask_rcnn_r50_fpn_1x_cityscapes_20200227-afe51d5a.pth \
    8  --format-only --options "txtfile_prefix=./mask_rcnn_cityscapes_test_results"
```

The generated png and txt would be under `./mask_rcnn_cityscapes_test_results` directory.

### Visualization

To see the SUNRGBD, ScanNet or KITTI points and detection results, you can run the following command

 ```bash
 python tools/test.py ${CONFIG_FILE} ${CKPT_PATH} --show --show-dir ${SHOW_DIR}
  ```

Aftering running this command, plotted results ***_points.obj and ***_pred.ply files in `${SHOW_DIR}`.

To see the points, detection results and ground truth of SUNRGBD, ScanNet or KITTI during evaluation time, you can run the following command
```bash
python tools/test.py ${CONFIG_FILE} ${CKPT_PATH} --eval 'mAP' --options "show=True" "out_dir=${SHOW_DIR}"
```
After running this command, you will obtain ***_points.ob, ***_pred.ply files and ***_gt.ply in `${SHOW_DIR}`.

You can use 3D visualization software such as the [MeshLab](http://www.meshlab.net/) to open the these files under `${SHOW_DIR}` to see the 3D detection output. Specifically, open `***_points.obj` to see the input point cloud and open `***_pred.ply` to see the predicted 3D bounding boxes. This allows the inference and results generation be done in remote server and the users can open them on their host with GUI.

**Notice**: The visualization API is a little unstable since we plan to refactor these parts together with MMDetection in the future.

### Point cloud demo

We provide a demo script to test a single sample.

```shell
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}]
```

Examples:

```shell
python demo/pcd_demo.py demo/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py \
    checkpoints/epoch_40.pth
```


### High-level APIs for testing images

#### Synchronous interface
Here is an example of building the model and test given images.

```python
from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = 'configs/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)
```

A notebook demo can be found in [demo/inference_demo.ipynb](https://github.com/open-mmlab/mmdetection/blob/master/demo/inference_demo.ipynb).

#### Asynchronous interface - supported for Python 3.7+

Async interface allows not to block CPU on GPU bound inference code and enables better CPU/GPU utilization for single threaded application. Inference can be done concurrently either between different input data samples or between different models of some inference pipeline.

See `tests/async_benchmark.py` to compare the speed of synchronous and asynchronous interfaces.

```python
import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.utils.contextmanagers import concurrent

async def main():
    config_file = 'configs/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    # test a single image and show the results
    img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once

    async with concurrent(streamqueue):
        result = await async_inference_detector(model, img)

    # visualize the results in a new window
    model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file='result.jpg')


asyncio.run(main())

```


## Train a model

MMDetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by adding the interval argument in the training config.
```python
evaluation = dict(interval=12)  # This evaluate the model per 12 epoch.
```

**\*Important\***: The default learning rate in config files is for 8 GPUs and 2 img/gpu (batch size = 8*2 = 16).
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.01 for 4 GPUs * 2 img/gpu and lr=0.08 for 16 GPUs * 4 img/gpu.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--no-validate` (**not suggested**): By default, the codebase will perform evaluation at every k (default value is 1, which can be modified like [this](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py#L174)) epochs during the training. To disable this behavior, use `--no-validate`.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

### Train with multiple machines

If you run MMDetection on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`. (This script also supports single machine training.)

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

Here is an example of using 16 GPUs to train Mask R-CNN on the dev partition.

```shell
GPUS=16 ./tools/slurm_train.sh dev mask_r50_1x configs/mask_rcnn_r50_fpn_1x_coco.py /nfs/xxxx/mask_rcnn_r50_fpn_1x
```

You can check [slurm_train.sh](https://github.com/open-mmlab/mmdetection/blob/master/tools/slurm_train.sh) for full arguments and environment variables.

If you have just multiple machines connected with ethernet, you can refer to
PyTorch [launch utility](https://pytorch.org/docs/stable/distributed_deprecated.html#launch-utility).
Usually it is slow if you do not have high speed networking like InfiniBand.

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

If you use launch training jobs with Slurm, you need to modify the config files (usually the 6th line from the bottom in config files) to set different communication ports.

In `config1.py`,
```python
dist_params = dict(backend='nccl', port=29500)
```

In `config2.py`,
```python
dist_params = dict(backend='nccl', port=29501)
```

Then you can launch two jobs with `config1.py` ang `config2.py`.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
```

## Useful tools

We provide lots of useful tools under `tools/` directory.

### Analyze logs

You can plot loss/mAP curves given a training log file. Run `pip install seaborn` first to install the dependency.

![loss curve image](../demo/loss_curve.png)

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

### Publish a model

Before you upload a model to AWS, you may want to
(1) convert model weights to CPU tensors, (2) delete the optimizer states and
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/faster_rcnn/latest.pth faster_rcnn_r50_fpn_1x_20190801.pth
```

The final output filename will be `faster_rcnn_r50_fpn_1x_20190801-{hash id}.pth`.

### Test the robustness of detectors

Please refer to [robustness_benchmarking.md](robustness_benchmarking.md).

### Convert to ONNX (experimental)

We provide a script to convert model to [ONNX](https://github.com/onnx/onnx) format. The converted model could be visualized by tools like [Netron](https://github.com/lutzroeder/netron).

```shell
python tools/pytorch2onnx.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${ONNX_FILE} [--shape ${INPUT_SHAPE}]
```

**Note**: This tool is still experimental. Customized operators are not supported for now. We set `use_torchvision=True` on-the-fly for `RoIPool` and `RoIAlign`.

## Tutorials

Currently, we provide four tutorials for users to [finetune models](tutorials/finetune.md), [add new dataset](tutorials/new_dataset.md), [design data pipeline](tutorials/data_pipeline.md) and [add new modules](tutorials/new_modules.md).
We also provide a full description about the [config system](config.md).
