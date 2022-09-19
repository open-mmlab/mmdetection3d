# 基准测试

这里我们对 MMDetection3D 和其他开源 3D 目标检测代码库中模型的训练速度和测试速度进行了基准测试。

## 配置

- 硬件：8 NVIDIA Tesla V100 (32G) GPUs, Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
- 软件：Python 3.7, CUDA 10.1, cuDNN 7.6.5, PyTorch 1.3, numba 0.48.0.
- 模型：由于不同代码库所实现的模型种类有所不同，在基准测试中我们选择了 SECOND、PointPillars、Part-A2 和 VoteNet 几种模型，分别与其他代码库中的相应模型实现进行了对比。
- 度量方法：我们使用整个训练过程中的平均吞吐量作为度量方法，并跳过每个 epoch 的前 50 次迭代以消除训练预热的影响。

## 主要结果

对于模型的训练速度（样本/秒），我们将 MMDetection3D 与其他实现了相同模型的代码库进行了对比。结果如下所示，表格内的数字越大，代表模型的训练速度越快。代码库中不支持的模型使用 `×` 进行标识。

|        模型         | MMDetection3D | OpenPCDet | votenet | Det3D |
| :-----------------: | :-----------: | :-------: | :-----: | :---: |
|       VoteNet       |      358      |     ×     |   77    |   ×   |
|  PointPillars-car   |      141      |     ×     |    ×    |  140  |
| PointPillars-3class |      107      |    44     |    ×    |   ×   |
|       SECOND        |      40       |    30     |    ×    |   ×   |
|       Part-A2       |      17       |    14     |    ×    |   ×   |

## 测试细节

### 为了计算速度所做的修改

- __MMDetection3D__：我们尝试使用与其他代码库中尽可能相同的配置，具体配置细节见 [基准测试配置](https://github.com/open-mmlab/MMDetection3D/blob/master/configs/benchmark)。

- __Det3D__：为了与 Det3D 进行比较，我们使用了 commit [519251e](https://github.com/poodarchu/Det3D/tree/519251e72a5c1fdd58972eabeac67808676b9bb7) 所对应的代码版本。

- __OpenPCDet__：为了与 OpenPCDet 进行比较，我们使用了 commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2) 所对应的代码版本。

  为了计算训练速度，我们在 `./tools/train_utils/train_utils.py` 文件中添加了用于记录运行时间的代码。我们对每个 epoch 的训练速度进行计算，并报告所有 epoch 的平均速度。

  <details>
    <summary>
    （为了使用相同方法进行测试所做的具体修改 - 点击展开）
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

- __MMDetection3D__：在 v0.1.0 版本下, 执行如下命令：

  ```bash
  ./tools/dist_train.sh configs/votenet/votenet_16x8_sunrgbd-3d-10class.py 8 --no-validate
  ```

- __votenet__：在 commit [2f6d6d3](https://github.com/facebookresearch/votenet/tree/2f6d6d36ff98d96901182e935afe48ccee82d566) 版本下，执行如下命令：

  ```bash
  python train.py --dataset sunrgbd --batch_size 16
  ```

  然后执行如下命令，对测试速度进行评估：

  ```bash
  python eval.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --batch_size 1 --dump_dir eval_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
  ```

  注意，为了计算推理速度，我们对 `eval.py` 进行了修改。

  <details>
  <summary>
  （为了对相同模型进行测试所做的具体修改 - 点击展开）
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

- __MMDetection3D__：在 v0.1.0 版本下, 执行如下命令：

  ```bash
  ./tools/dist_train.sh configs/benchmark/hv_pointpillars_secfpn_3x8_100e_det3d_kitti-3d-car.py 8 --no-validate
  ```

- __Det3D__：在 commit [519251e](https://github.com/poodarchu/Det3D/tree/519251e72a5c1fdd58972eabeac67808676b9bb7) 版本下，使用 `kitti_point_pillars_mghead_syncbn.py` 并执行如下命令：

  ```bash
  ./tools/scripts/train.sh --launcher=slurm --gpus=8
  ```

  注意，为了训练 PointPillars，我们对 `train.sh` 进行了修改。

  <details>
  <summary>
  （为了对相同模型进行测试所做的具体修改 - 点击展开）
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

- __MMDetection3D__：在 v0.1.0 版本下, 执行如下命令：

  ```bash
  ./tools/dist_train.sh configs/benchmark/hv_pointpillars_secfpn_4x8_80e_pcdet_kitti-3d-3class.py 8 --no-validate
  ```

- __OpenPCDet__：在 commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2) 版本下，执行如下命令：

  ```bash
  cd tools
  sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} 8  --cfg_file ./cfgs/kitti_models/pointpillar.yaml --batch_size 32  --workers 32 --epochs 80
  ```

### SECOND

基准测试中的 SECOND 指在 [second.Pytorch](https://github.com/traveller59/second.pytorch) 首次被实现的 [SECONDv1.5](https://github.com/traveller59/second.pytorch/blob/master/second/configs/all.fhd.config)。Det3D 实现的 SECOND 中，使用了自己实现的 Multi-Group Head，因此无法将它的速度与其他代码库进行对比。

- __MMDetection3D__：在 v0.1.0 版本下, 执行如下命令：

  ```bash
  ./tools/dist_train.sh configs/benchmark/hv_second_secfpn_4x8_80e_pcdet_kitti-3d-3class.py 8 --no-validate
  ```

- __OpenPCDet__：在 commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2) 版本下，执行如下命令：

  ```bash
  cd tools
  sh ./scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} 8  --cfg_file ./cfgs/kitti_models/second.yaml --batch_size 32  --workers 32 --epochs 80
  ```

### Part-A2

- __MMDetection3D__：在 v0.1.0 版本下, 执行如下命令：

  ```bash
  ./tools/dist_train.sh configs/benchmark/hv_PartA2_secfpn_4x8_cyclic_80e_pcdet_kitti-3d-3class.py 8 --no-validate
  ```

- __OpenPCDet__：在 commit [b32fbddb](https://github.com/open-mmlab/OpenPCDet/tree/b32fbddbe06183507bad433ed99b407cbc2175c2) 版本下，执行如下命令以进行模型训练：

  ```bash
  cd tools
  sh ./scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} 8  --cfg_file ./cfgs/kitti_models/PartA2.yaml --batch_size 32 --workers 32 --epochs 80
  ```
