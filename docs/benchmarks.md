
# Benchmarks

Here we benchmark the training and testing speed of models in MMDetection3D,
with some other popular open source 3D detection codebases.


## Settings

* Hardwares: 8 NVIDIA Tesla V100 (32G) GPUs, Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
* Software: Python 3.7, CUDA 10.1, cuDNN 7.6.5, PyTorch 1.3, numba 0.48.0.
* Model: Since all the other codebases implements different models, we compare the corresponding models with them separately. We try to use as similar settings as those of other codebases as possible using [benchmark configs](https://github.com/open-mmlab/MMDetection3D/blob/master/configs/benchmark).
* Metrics: We use the average throughput in iterations of the entire training run and skip the first 50 iterations of each epoch to skip GPU warmup time.
  Note that the throughput of a detector typically changes during training, because it depends on the predictions of the model.


## Main Results

### VoteNet

We compare our implementation with VoteNet and report the performance of VoteNets on SUNRGB-D v2 dataset under the AP@0.5 metric.

```eval_rst
  +----------------+---------------------+--------------------+-------------------+--------+
  | Implementation | Training (sample/s) | Testing (sample/s) | Training Time (h) | AP@0.5 |
  +================+=====================+====================+===================+========+
  | MMDetection3D  |                     |                    |                   |        |
  +----------------+---------------------+--------------------+-------------------+--------+
  | VoteNet        |                     |                    |                   |        |
  +----------------+---------------------+--------------------+-------------------+--------+
```

### PointPillars

Since Det3D only provides PointPillars on car class while PCDet only provides PointPillars
on 3 classes, we compare with them separately. For performance on single class, we report the AP on moderate
condition following the KITTI benchmark and compare average AP over all classes on moderate condition for
performance on 3 classes.

```eval_rst
  +----------------+---------------------+--------------------+-------------------+-------------+
  | Implementation | Training (sample/s) | Testing (sample/s) | Training Time (h) | Moderate AP |
  +================+=====================+====================+===================+=============+
  | MMDetection3D  |                     |                    |                   |             |
  +----------------+---------------------+--------------------+-------------------+-------------+
  | PCDet          |                     |                    |                   |             |
  +----------------+---------------------+--------------------+-------------------+-------------+
```

```eval_rst
  +----------------+---------------------+--------------------+-------------------+-------------+
  | Implementation | Training (sample/s) | Testing (sample/s) | Training Time (h) | Moderate AP |
  +================+=====================+====================+===================+=============+
  | MMDetection3D  |                     |                    |                   |             |
  +----------------+---------------------+--------------------+-------------------+-------------+
  | Det3D          |                     |                    |                   |             |
  +----------------+---------------------+--------------------+-------------------+-------------+
```

### SECOND

Det3D provides a different SECOND on car class and we cannot train the original SECOND by modifying the config.
So we only compare with PCDet, which is a SECOND model on 3 classes, we report the AP on moderate
condition following the KITTI benchmark and compare average AP over all classes on moderate condition for
performance on 3 classes.

  ```eval_rst
    +----------------+---------------------+--------------------+-------------------+-------------+
    | Implementation | Training (sample/s) | Testing (sample/s) | Training Time (h) | Moderate AP |
    +================+=====================+====================+===================+=============+
    | MMDetection3D  |                     |                    |                   |             |
    +----------------+---------------------+--------------------+-------------------+-------------+
    | PCDet          |                     |                    |                   |             |
    +----------------+---------------------+--------------------+-------------------+-------------+
  ```

### Part-A2

We benchmark Part-A2 with that in PCDet. We report the AP on moderate condition following the KITTI benchmark
and compare average AP over all classes on moderate condition for performance on 3 classes.

  ```eval_rst
    +----------------+---------------------+--------------------+-------------------+-------------+
    | Implementation | Training (sample/s) | Testing (sample/s) | Training Time (h) | Moderate AP |
    +================+=====================+====================+===================+=============+
    | MMDetection3D  |                     |                    |                   |             |
    +----------------+---------------------+--------------------+-------------------+-------------+
    | PCDet          |                     |                    |                   |             |
    +----------------+---------------------+--------------------+-------------------+-------------+
  ```

## Details of Comparison

### VoteNet

* __MMDetection3D__: With release v0.1.0, run
```
./tools/dist_train.sh configs/votenet/mask_rcnn_r50_caffe_fpn_1x_coco.py 8
```
* __votenet__:


### PointPillars

* __MMDetection3D__: With release v0.1.0, run
```
```
* __PCDet__: At commit xxxx


### SECOND

* __MMDetection3D__: With release v0.1.0, run
```
```

* __PCDet__:


### Part-A2

* __MMDetection3D__: With release v0.1.0, run
```
```

* __PCDet__: At commit xxxx

### Modification for Calculating Training Speed
