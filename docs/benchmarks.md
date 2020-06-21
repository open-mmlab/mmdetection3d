
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

We compare our implementation of VoteNet with [votenet](https://github.com/facebookresearch/votenet/) and report the performance on SUNRGB-D v2 dataset under the AP@0.5 metric.

```eval_rst
  +----------------+---------------------+--------------------+--------+
  | Implementation | Training (sample/s) | Testing (sample/s) | AP@0.5 |
  +================+=====================+====================+========+
  | MMDetection3D  |        358          |         17         |  35.8  |
  +----------------+---------------------+--------------------+--------+
  | VoteNet        |        77           |         3          |  31.5  |
  +----------------+---------------------+--------------------+--------+
```

### PointPillars

Since [Det3D](https://github.com/poodarchu/Det3D/) only provides PointPillars on car class while [PCDet](https://github.com/sshaoshuai/PCDet) only provides PointPillars
on 3 classes, we compare with them separately. For performance on single class, we report the AP on moderate
condition following the KITTI benchmark and compare average AP over all classes on moderate condition for
performance on 3 classes.

```eval_rst
  +----------------+---------------------+--------------------+
  | Implementation | Training (sample/s) | Testing (sample/s) |
  +================+=====================+====================+
  | MMDetection3D  |         141         |                    |
  +----------------+---------------------+--------------------+
  | Det3D          |         140         |        20          |
  +----------------+---------------------+--------------------+
```

```eval_rst
  +----------------+---------------------+--------------------+
  | Implementation | Training (sample/s) | Testing (sample/s) |
  +================+=====================+====================+
  | MMDetection3D  |         120         |                    |
  +----------------+---------------------+--------------------+
  | PCDet          |         43          |        64          |
  +----------------+---------------------+--------------------+
```

### SECOND

[Det3D](https://github.com/poodarchu/Det3D/) provides a different SECOND on car class and we cannot train the original SECOND by modifying the config.
So we only compare with [PCDet](https://github.com/sshaoshuai/PCDet), which is a SECOND model on 3 classes, we report the AP on moderate
condition following the KITTI benchmark and compare average AP over all classes on moderate condition for
performance on 3 classes.

  ```eval_rst
    +----------------+---------------------+--------------------+
    | Implementation | Training (sample/s) | Testing (sample/s) |
    +================+=====================+====================+
    | MMDetection3D  |         54          |                    |
    +----------------+---------------------+--------------------+
    | PCDet          |         44          |         30         |
    +----------------+---------------------+--------------------+
  ```

### Part-A2

We benchmark Part-A2 with that in [PCDet](https://github.com/sshaoshuai/PCDet). We report the AP on moderate condition following the KITTI benchmark
and compare average AP over all classes on moderate condition for performance on 3 classes.

  ```eval_rst
    +----------------+---------------------+--------------------+
    | Implementation | Training (sample/s) | Testing (sample/s) |
    +================+=====================+====================+
    | MMDetection3D  |         17          |                    |
    +----------------+---------------------+--------------------+
    | PCDet          |         15          |         12         |
    +----------------+---------------------+--------------------+
  ```

## Details of Comparison

### Modification for Calculating Speed

* __Det3D__: At commit 255c593


* __PCDet__: At commit 2244be4



### VoteNet

* __MMDetection3D__: With release v0.1.0, run
```
./tools/dist_train.sh configs/votenet/votenet_16x8_sunrgbd-3d-10class.py 8 --no-validate
```
* __votenet__: At commit xxxx, run
```
```


### PointPillars

* __MMDetection3D__: With release v0.1.0, run
```
./tools/dist_train.sh configs/benchmark/hv_pointpillars_secfpn_6x8_160e_pcdet_kitti-3d-3class.py 8 --no-validate
```
* __PCDet__: At commit xxxx
```
./tools/dist_train.sh configs/benchmark/hv_pointpillars_secfpn_6x8_160e_pcdet_kitti-3d-3class.py 8 --no-validate
```

* __MMDetection3D__: With release v0.1.0, run
```
./tools/dist_train.sh configs/benchmark/hv_pointpillars_secfpn_3x8_100e_det3d_kitti-3d-car.py 8 --no-validate
```
* __Det3D__: At commit xxxx
```

```


### SECOND

* __MMDetection3D__: With release v0.1.0, run
```
./tools/dist_train.sh configs/benchmark/hv_second_secfpn_6x8_80e_pcdet_kitti-3d-3class.py 8 --no-validate
```

* __PCDet__: At commit 2244be4


### Part-A2

* __MMDetection3D__: With release v0.1.0, run
```
./tools/dist_train.sh configs/benchmark/hv_PartA2_secfpn_2x8_cyclic_80e_pcdet_kitti-3d-3class.py 8 --no-validate
```

* __PCDet__:
