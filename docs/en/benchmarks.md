# Benchmarks

Here we benchmark the training and testing speed of models in MMDetection3D,
with some other open source 3D detection codebases.

## Settings

- Hardwares: 8 NVIDIA Tesla V100 (32G) GPUs, Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
- Software: Python 3.7, CUDA 10.1, cuDNN 7.6.5, PyTorch 1.3, numba 0.48.0.
- Model: Since all the other codebases implements different models, we compare the corresponding models including SECOND, PointPillars, Part-A2, and VoteNet with them separately.
- Metrics: We use the average throughput in iterations of the entire training run and skip the first 50 iterations of each epoch to skip GPU warmup time.

## Main Results

We compare the training speed (samples/s) with other codebases if they implement the similar models. The results are as below, the greater the numbers in the table, the faster of the training process. The models that are not supported by other codebases are marked by `×`.

|       Methods       | MMDetection3D | OpenPCDet | votenet | Det3D |
| :-----------------: | :-----------: | :-------: | :-----: | :---: |
|       VoteNet       |      358      |     ×     |   77    |   ×   |
|  PointPillars-car   |      141      |     ×     |    ×    |  140  |
| PointPillars-3class |      107      |    44     |    ×    |   ×   |
|       SECOND        |      40       |    30     |    ×    |   ×   |
|       Part-A2       |      17       |    14     |    ×    |   ×   |

## Details of Comparison

### Modification for Calculating Speed

- __MMDetection3D__: We try to use as similar settings as those of other codebases as possible using [benchmark configs](https://github.com/open-mmlab/MMDetection3D/blob/master/configs/benchmark).

- __Det3D__: For comparison with Det3D, we use the commit [519251e](https://github.com/poodarchu/Det3D/tree/519251e72a5c1fdd58972eabeac67808676b9bb7).

- __OpenPCDet__: For comparison with OpenPCDet, we use the commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2).

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

### VoteNet

- __MMDetection3D__: With release v0.1.0, run

  ```bash
  ./tools/dist_train.sh configs/votenet/votenet_16x8_sunrgbd-3d-10class.py 8 --no-validate
  ```

- __votenet__: At commit [2f6d6d3](https://github.com/facebookresearch/votenet/tree/2f6d6d36ff98d96901182e935afe48ccee82d566), run

  ```bash
  python train.py --dataset sunrgbd --batch_size 16
  ```

  Then benchmark the test speed by running

  ```bash
  python eval.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --batch_size 1 --dump_dir eval_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
  ```

  Note that eval.py is modified to compute inference time.

  <details>
  <summary>
  (diff to benchmark the similar models - click to expand)
  </summary>

  ```diff
  diff --git a/eval.py b/eval.py
    index c0b2886..04921e9 100644
    --- a/eval.py
    +++ b/eval.py
    @@ -10,6 +10,7 @@ import os
     import sys
     import numpy as np
     from datetime import datetime
    +import time
     import argparse
     import importlib
     import torch
    @@ -28,7 +29,7 @@ parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint pa
     parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
     parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
     parser.add_argument('--num_target', type=int, default=256, help='Point Number [default: 256]')
    -parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
    +parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 8]')
     parser.add_argument('--vote_factor', type=int, default=1, help='Number of votes generated from each seed [default: 1]')
     parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
     parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
    @@ -132,6 +133,7 @@ CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms': FLAGS.
     # ------------------------------------------------------------------------- GLOBAL CONFIG END

     def evaluate_one_epoch():
    +    time_list = list()
         stat_dict = {}
         ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
             for iou_thresh in AP_IOU_THRESHOLDS]
    @@ -144,6 +146,8 @@ def evaluate_one_epoch():

             # Forward pass
             inputs = {'point_clouds': batch_data_label['point_clouds']}
    +        torch.cuda.synchronize()
    +        start_time = time.perf_counter()
             with torch.no_grad():
                 end_points = net(inputs)

    @@ -161,6 +165,12 @@ def evaluate_one_epoch():

             batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
             batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
    +        torch.cuda.synchronize()
    +        elapsed = time.perf_counter() - start_time
    +        time_list.append(elapsed)
    +
    +        if len(time_list==200):
    +            print("average inference time: %4f"%(sum(time_list[5:])/len(time_list[5:])))
             for ap_calculator in ap_calculator_list:
                 ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

  ```

### PointPillars-car

- __MMDetection3D__: With release v0.1.0, run

  ```bash
  ./tools/dist_train.sh configs/benchmark/hv_pointpillars_secfpn_3x8_100e_det3d_kitti-3d-car.py 8 --no-validate
  ```

- __Det3D__: At commit [519251e](https://github.com/poodarchu/Det3D/tree/519251e72a5c1fdd58972eabeac67808676b9bb7), use `kitti_point_pillars_mghead_syncbn.py` and run

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

### PointPillars-3class

- __MMDetection3D__: With release v0.1.0, run

  ```bash
  ./tools/dist_train.sh configs/benchmark/hv_pointpillars_secfpn_4x8_80e_pcdet_kitti-3d-3class.py 8 --no-validate
  ```

- __OpenPCDet__: At commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2), run

  ```bash
  cd tools
  sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} 8  --cfg_file ./cfgs/kitti_models/pointpillar.yaml --batch_size 32  --workers 32 --epochs 80
  ```

### SECOND

For SECOND, we mean the [SECONDv1.5](https://github.com/traveller59/second.pytorch/blob/master/second/configs/all.fhd.config) that was first implemented in [second.Pytorch](https://github.com/traveller59/second.pytorch). Det3D's implementation of SECOND uses its self-implemented Multi-Group Head, so its speed is not compatible with other codebases.

- __MMDetection3D__: With release v0.1.0, run

  ```bash
  ./tools/dist_train.sh configs/benchmark/hv_second_secfpn_4x8_80e_pcdet_kitti-3d-3class.py 8 --no-validate
  ```

- __OpenPCDet__: At commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2), run

  ```bash
  cd tools
  sh ./scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} 8  --cfg_file ./cfgs/kitti_models/second.yaml --batch_size 32  --workers 32 --epochs 80
  ```

### Part-A2

- __MMDetection3D__: With release v0.1.0, run

  ```bash
  ./tools/dist_train.sh configs/benchmark/hv_PartA2_secfpn_4x8_cyclic_80e_pcdet_kitti-3d-3class.py 8 --no-validate
  ```

- __OpenPCDet__: At commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2), train the model by running

  ```bash
  cd tools
  sh ./scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} 8  --cfg_file ./cfgs/kitti_models/PartA2.yaml --batch_size 32 --workers 32 --epochs 80
  ```
