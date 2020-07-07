
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
| votenet_       |        77           |         3          |  31.5  |
+----------------+---------------------+--------------------+--------+

```

### PointPillars

Since [Det3D](https://github.com/poodarchu/Det3D/) only provides PointPillars on car class while [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2) only provides PointPillars
on 3 classes, we compare with them separately. For performance on single class, we report the AP on moderate
condition following the KITTI benchmark and compare average AP over all classes on moderate condition for
performance on 3 classes.

```eval_rst
+----------------+---------------------+--------------------+
| Implementation | Training (sample/s) | Testing (sample/s) |
+================+=====================+====================+
| MMDetection3D  |         141         |       44.3         |
+----------------+---------------------+--------------------+
| Det3D          |         140         |        20          |
+----------------+---------------------+--------------------+
```

```eval_rst
  +----------------+---------------------+--------------------+
  | Implementation | Training (sample/s) | Testing (sample/s) |
  +================+=====================+====================+
  | MMDetection3D  |         107         |        45          |
  +----------------+---------------------+--------------------+
  | OpenPCDet      |         44          |        25          |
  +----------------+---------------------+--------------------+
```

### SECOND

[Det3D](https://github.com/poodarchu/Det3D/) provides a different SECOND on car class and we cannot train the original SECOND by modifying the config.
So we only compare with [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2), which is a SECOND model on 3 classes, we report the AP on moderate
condition following the KITTI benchmark and compare average AP over all classes on moderate condition for
performance on 3 classes.

  ```eval_rst
    +----------------+---------------------+--------------------+
    | Implementation | Training (sample/s) | Testing (sample/s) |
    +================+=====================+====================+
    | MMDetection3D  |         40          |         27         |
    +----------------+---------------------+--------------------+
    | OpenPCDet      |         30          |         32         |
    +----------------+---------------------+--------------------+
  ```

### Part-A2

We benchmark Part-A2 with that in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2). We report the AP on moderate condition following the KITTI benchmark
and compare average AP over all classes on moderate condition for performance on 3 classes.

  ```eval_rst
    +----------------+---------------------+--------------------+
    | Implementation | Training (sample/s) | Testing (sample/s) |
    +================+=====================+====================+
    | MMDetection3D  |         17          |         11         |
    +----------------+---------------------+--------------------+
    | OpenPCDet      |         14          |         13         |
    +----------------+---------------------+--------------------+
  ```

## Details of Comparison

### Modification for Calculating Speed

* __Det3D__: At commit 255c593

* __OpenPCDet__: At commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2)

    For training speed, we add code to record the running time in the file `./tools/train_utils/train_utils.py`. We calculate the speed of each epoch, and report the average speed of all the epochs.
    <details>
    <summary>
    (diff to make it use the same method for benchmarking speed - click to expand)
    </summary>

    ```diff
    diff --git a/tools/train_utils/train_utils.py b/tools/train_utils/train_utils.py
    index 91f21dd..021359d 100644
    --- a/tools/train_utils/train_utils.py
    +++ b/tools/train_utils/train_utils.py
    @@ -2,6 +2,7 @@ import torch
     import os
     import glob
     import tqdm
    +import datetime
     from torch.nn.utils import clip_grad_norm_


    @@ -13,7 +14,10 @@ def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, ac
         if rank == 0:
             pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    +    start_time = None
         for cur_it in range(total_it_each_epoch):
    +        if cur_it > 49 and start_time is None:
    +            start_time = datetime.datetime.now()
             try:
                 batch = next(dataloader_iter)
             except StopIteration:
    @@ -55,9 +59,11 @@ def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, ac
                     tb_log.add_scalar('learning_rate', cur_lr, accumulated_iter)
                     for key, val in tb_dict.items():
                         tb_log.add_scalar('train_' + key, val, accumulated_iter)
    +    endtime = datetime.datetime.now()
    +    speed = (endtime - start_time).seconds / (total_it_each_epoch - 50)
         if rank == 0:
             pbar.close()
    -    return accumulated_iter
    +    return accumulated_iter, speed


     def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
    @@ -65,6 +71,7 @@ def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_
                     lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                     merge_all_iters_to_one_epoch=False):
         accumulated_iter = start_iter
    +    speeds = []
         with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
             total_it_each_epoch = len(train_loader)
             if merge_all_iters_to_one_epoch:
    @@ -82,7 +89,7 @@ def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_
                     cur_scheduler = lr_warmup_scheduler
                 else:
                     cur_scheduler = lr_scheduler
    -            accumulated_iter = train_one_epoch(
    +            accumulated_iter, speed = train_one_epoch(
                     model, optimizer, train_loader, model_func,
                     lr_scheduler=cur_scheduler,
                     accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
    @@ -91,7 +98,7 @@ def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_
                     total_it_each_epoch=total_it_each_epoch,
                     dataloader_iter=dataloader_iter
                 )
    -
    +            speeds.append(speed)
                 # save trained model
                 trained_epoch = cur_epoch + 1
                 if trained_epoch % ckpt_save_interval == 0 and rank == 0:
    @@ -107,6 +114,8 @@ def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_
                     save_checkpoint(
                         checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                     )
    +            print(speed)
    +    print(f'*******{sum(speeds) / len(speeds)}******')


     def model_state_to_cpu(model_state):
    ```

    </details>

    For testing speed, we add code to record the running time in the file `./tools/eval_utils/eval_utils.py`.
    <details>
    <summary>
    (diff to make it use the same method for benchmarking speed - click to expand)
    </summary>

    ```diff
    diff --git a/tools/eval_utils/eval_utils.py b/tools/eval_utils/eval_utils.py
    index 0cbf17b..f51e687 100644
    --- a/tools/eval_utils/eval_utils.py
    +++ b/tools/eval_utils/eval_utils.py
    @@ -49,8 +49,11 @@ def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, sa

         if cfg.LOCAL_RANK == 0:
             progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    -    start_time = time.time()
    +    num_warmup = 5
    +    pure_inf_time = 0
         for i, batch_dict in enumerate(dataloader):
    +        torch.cuda.synchronize()
    +        start_time = time.perf_counter()
             for key, val in batch_dict.items():
                 if not isinstance(val, np.ndarray):
                     continue
    @@ -61,7 +64,14 @@ def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, sa
             with torch.no_grad():
                 pred_dicts, ret_dict = model(batch_dict)
             disp_dict = {}
    -
    +        torch.cuda.synchronize()
    +        elapsed = time.perf_counter() - start_time
    +        if i >= num_warmup:
    +            pure_inf_time += elapsed
    +        if (i + 1) == 2000:
    +            pure_inf_time += elapsed
    +            fps = (i + 1 - num_warmup) / pure_inf_time
    +            out_str = f'Overall fps: {fps:.1f} img / s'
             statistics_info(cfg, ret_dict, metric, disp_dict)
             annos = dataset.generate_prediction_dicts(
                 batch_dict, pred_dicts, class_names,
    @@ -71,7 +81,7 @@ def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, sa
             if cfg.LOCAL_RANK == 0:
                 progress_bar.set_postfix(disp_dict)
                 progress_bar.update()
    -
    +    print(out_str)
         if cfg.LOCAL_RANK == 0:
             progress_bar.close()
    ```

    </details>

### VoteNet

* __MMDetection3D__: With release v0.1.0, run

  ```bash
  ./tools/dist_train.sh configs/votenet/votenet_16x8_sunrgbd-3d-10class.py 8 --no-validate
  ```

* __votenet__: At commit 2f6d6d3, run

  ```bash
  python train.py --dataset sunrgbd --batch_size 16
  ```

### PointPillars

* __MMDetection3D__: With release v0.1.0, run

  ```bash
  ./tools/dist_train.sh configs/benchmark/hv_pointpillars_secfpn_4x8_80e_pcdet_kitti-3d-3class.py 8 --no-validate
  ```

* __OpenPCDet__: At commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2)

  ```bash
  cd tools
  sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} 8  --cfg_file ./cfgs/pointpillar.yaml --batch_size 32  --workers 32
  ```

* __MMDetection3D__: With release v0.1.0, run

  ```bash
  /tools/dist_train.sh configs/benchmark/hv_pointpillars_secfpn_3x8_100e_det3d_kitti-3d-car.py 8 --no-validate
  ```

* __Det3D__: At commit 255c593, use kitti_point_pillars_mghead_syncbn.py and run

  ```bash
  ./tools/scripts/train.sh --launcher=slurm --gpus=8
  ```

  Note that the config in train.sh is modified to train point pillars.

  <details>
  <summary>
  (diff to benchmark the similar models - click to expand)
  </summary>

  ```diff
  diff --git a/tools/scripts/train.sh b/tools/scripts/train.sh
  index 3a93f95..461e0ea 100755
  --- a/tools/scripts/train.sh
  +++ b/tools/scripts/train.sh
  @@ -16,9 +16,9 @@ then
   fi

   # Voxelnet
  -python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/  kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py --work_dir=$SECOND_WORK_DIR
  +# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/  kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py --work_dir=$SECOND_WORK_DIR
   # python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/cbgs/configs/  nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$NUSC_CBGS_WORK_DIR
   # python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/  lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$LYFT_CBGS_WORK_DIR

   # PointPillars
  -# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py ./examples/point_pillars/configs/  original_pp_mghead_syncbn_kitti.py --work_dir=$PP_WORK_DIR
  +python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py ./examples/point_pillars/configs/  kitti_point_pillars_mghead_syncbn.py
  ```

  </details>

### SECOND

* __MMDetection3D__: With release v0.1.0, run

  ```bash
  ./tools/dist_train.sh configs/benchmark/hv_second_secfpn_4x8_80e_pcdet_kitti-3d-3class.py 8 --no-validate
  ```

* __OpenPCDet__: At commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2)

  ```bash
  cd tools
  sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} 8  --cfg_file ./cfgs/second.yaml --batch_size 32  --workers 32
  ```

### Part-A2

* __MMDetection3D__: With release v0.1.0, run

  ```bash
  ./tools/dist_train.sh configs/benchmark/hv_PartA2_secfpn_4x8_cyclic_80e_pcdet_kitti-3d-3class.py 8 --no-validate
  ```

* __OpenPCDet__: At commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2)

  ```bash
  cd tools
  sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} 8  --cfg_file ./cfgs/PartA2.yaml --batch_size 32 --workers 32
  ```
