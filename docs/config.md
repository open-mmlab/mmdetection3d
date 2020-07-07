# Config System
We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/print_config.py /PATH/TO/CONFIG` to see the complete config.
You may also pass `--options xxx.yyy=zzz` to see updated config.

## Config File Structure

There are 4 basic component types under `config/_base_`, dataset, model, schedule, default_runtime.
Many methods could be easily constructed with one of each like Faster R-CNN, Mask R-CNN, Cascade R-CNN, RPN, SSD.
The configs that are composed by components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from exiting methods.
For example, if some modification is made base on Faster R-CNN, user may first inherit the basic Faster R-CNN structure by specifying `_base_ = ../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods, you may create a folder `xxx_rcnn` under `configs`,

Please refer to [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#config) for detailed documentation.

## Config Name Style

We follow the below style to name config files. Contributors are advised to follow the same style.

```
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
```

`{xxx}` is required field and `[yyy]` is optional.

- `{model}`: model type like `faster_rcnn`, `mask_rcnn`, etc.
- `[model setting]`: specific setting for some model, like `without_semantic` for `htc`, `moment` for `reppoints`, etc.
- `{backbone}`: backbone type like `r50` (ResNet-50), `x101` (ResNeXt-101).
- `{neck}`: neck type like `fpn`, `pafpn`, `nasfpn`, `c4`.
- `[norm_setting]`: `bn` (Batch Normalization) is used unless specified, other norm layer type could be `gn` (Group Normalization), `syncbn` (Synchronized Batch Normalization).
`gn-head`/`gn-neck` indicates GN is applied in head/neck only, while `gn-all` means GN is applied in the entire model, e.g. backbone, neck, head.
- `[misc]`: miscellaneous setting/plugins of model, e.g. `dconv`, `gcb`, `attention`, `albu`, `mstrain`.
- `[gpu x batch_per_gpu]`: GPUs and samples per GPU, `8x2` is used by default.
- `{schedule}`: training schedule, options are `1x`, `2x`, `20e`, etc.
`1x` and `2x` means 12 epochs and 24 epochs respectively.
`20e` is adopted in cascade models, which denotes 20 epochs.
For `1x`/`2x`, initial learning rate decays by a factor of 10 at the 8/16th and 11/22th epochs.
For `20e`, initial learning rate decays by a factor of 10 at the 16th and 19th epochs.
- `{dataset}`: dataset like `coco`, `cityscapes`, `voc_0712`, `wider_face`.

## FAQ

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some of fields in base configs.
You may refer to [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#inherit-from-base-config-with-ignored-fields) for simple inllustration.

In MMDetection or MMDetection3D, for example, to change the backbone of Mask R-CNN with the following config.

```python
model = dict(
    type='MaskRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(...),
    rpn_head=dict(...),
    roi_head=dict(...))
```

`ResNet` and `HRNet` use different keywords to construct.

```python
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    neck=dict(...))
```

The `_delete_=True` would replace all old keys in `backbone` field with new keys new keys.

### Use intermediate variables in configs

Some intermediate variables are used in the configs files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, user need to pass the intermediate variables into corresponding fields again.
For example, we would like to use multi scale strategy to train a Mask R-CNN. `train_pipeline`/`test_pipeline` are intermediate variable we would like modify.
```python
_base_ = './mask_rcnn_r50_fpn_1x_coco.py'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode="value",
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
```
We first define the new `train_pipeline`/`test_pipeline` and pass them into `data`.


## An example of VoteNet

```python
model = dict(
    type='VoteNet',  # The type of detector, refer to mmdet3d.models.detectors for more details
    backbone=dict(
        type='PointNet2SASSG',  # The type of the backbone， refer to mmdet3d.models.backbones for more details
        in_channels=4,  # Input channels of point cloud
        num_points=(2048, 1024, 512, 256),  # The number of points which each SA module samples
        radius=(0.2, 0.4, 0.8, 1.2),  # Radius for each set abstraction layer
        num_samples=(64, 32, 16, 16),  # Number of samples for each set abstraction layer
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),  # Out channels of each mlp in SA module
        fp_channels=((256, 256), (256, 256)),  # Out channels of each mlp in FP module
        norm_cfg=dict(type='BN2d'),  # Config of normalization layer
        pool_mod='max'),  # Pool method ('max' or 'avg') for SA modules
    bbox_head=dict(
        type='VoteHead',  # The type of bbox head, refer to mmdet3d.models.dense_heads for more details
        num_classes=18,  # Number of classes for classification
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',  # The type of bbox_coder, refer to mmdet3d.core.bbox.coders for more details
            num_sizes=18,  # Number of size clusters
            num_dir_bins=1,   # Number of bins to encode direction angle
            with_rot=False,  # Whether the bbox is with rotation
            mean_sizes=[[0.76966727, 0.8116021, 0.92573744],
                        [1.876858, 1.8425595, 1.1931566],
                        [0.61328, 0.6148609, 0.7182701],
                        [1.3955007, 1.5121545, 0.83443564],
                        [0.97949594, 1.0675149, 0.6329687],
                        [0.531663, 0.5955577, 1.7500148],
                        [0.9624706, 0.72462326, 1.1481868],
                        [0.83221924, 1.0490936, 1.6875663],
                        [0.21132214, 0.4206159, 0.5372846],
                        [1.4440073, 1.8970833, 0.26985747],
                        [1.0294262, 1.4040797, 0.87554324],
                        [1.3766412, 0.65521795, 1.6813129],
                        [0.6650819, 0.71111923, 1.298853],
                        [0.41999173, 0.37906948, 1.7513971],
                        [0.59359556, 0.5912492, 0.73919016],
                        [0.50867593, 0.50656086, 0.30136237],
                        [1.1511526, 1.0546296, 0.49706793],
                        [0.47535285, 0.49249494, 0.5802117]]),  # Mean sizes for each class, the order is consistent with class_names.
        vote_moudule_cfg=dict(  # Config to vote module branch, refer to mmdet3d.models.model_utils for more details
            in_channels=256,  # Input channels for vote_module
            vote_per_seed=1,  # Number of votes to generate for each seed
            gt_per_seed=3,  # Number of gts for each seed
            conv_channels=(256, 256),  # Channels for convolution
            conv_cfg=dict(type='Conv1d'),  # Config to convolution
            norm_cfg=dict(type='BN1d'),  # Config to normalization
            norm_feats=True,  # Whether to normalize features
            vote_loss=dict(  # Config to the loss function for voting branch
                type='ChamferDistance',  # Type of loss for voting branch
                mode='l1',  # Loss mode of voting branch
                reduction='none',  # Specifies the reduction to apply to the output
                loss_dst_weight=10.0)),  # Destination loss weight of the voting branch
        vote_aggregation_cfg=dict(  # Config to vote aggregation branch
            num_point=256,  # Number of points for the set abstraction layer in vote aggregation branch
            radius=0.3,  # Radius for the set abstraction layer in vote aggregation branch
            num_sample=16,  # Number of samples for the set abstraction layer in vote aggregation branch
            mlp_channels=[256, 128, 128, 128],  # Mlp channels for the set abstraction layer in vote aggregation branch
            use_xyz=True,  # Whether to use xyz
            normalize_xyz=True),  # Whether to normalize xyz
        feat_channels=(128, 128),  # Channels for feature convolution
        conv_cfg=dict(type='Conv1d'),  # Config to convolution
        norm_cfg=dict(type='BN1d'),  # Config to normalization
        objectness_loss=dict(  # Config to objectness loss
            type='CrossEntropyLoss',  # Type of loss
            class_weight=[0.2, 0.8],  # Class weight of the objectness loss
            reduction='sum',  # Specifies the reduction to apply to the output
            loss_weight=5.0),  # Loss weight of the objectness loss
        center_loss=dict(  # Config to center loss
            type='ChamferDistance',  # Type of loss
            mode='l2',  # Loss mode of center loss
            reduction='sum',  # Specifies the reduction to apply to the output
            loss_src_weight=10.0,  # Source loss weight of the voting branch.
            loss_dst_weight=10.0),  # Destination loss weight of the voting branch.
        dir_class_loss=dict(  # Config to direction classification loss
            type='CrossEntropyLoss',  # Type of loss
            reduction='sum',  # Specifies the reduction to apply to the output
            loss_weight=1.0),  # Loss weight of the direction classification loss
        dir_res_loss=dict(  # Config to direction residual loss
            type='SmoothL1Loss',  # Type of loss
            reduction='sum',  # Specifies the reduction to apply to the output
            loss_weight=10.0),  # Loss weight of the direction residual loss
        size_class_loss=dict(  # Config to size classification loss
            type='CrossEntropyLoss',  # Type of loss
            reduction='sum',  # Specifies the reduction to apply to the output
            loss_weight=1.0),  # Loss weight of the size classification loss
        size_res_loss=dict(  # Config to size residual loss
            type='SmoothL1Loss',  # Type of loss
            reduction='sum',  # Specifies the reduction to apply to the output
            loss_weight=3.3333333333333335),  # Loss weight of the size residual loss
        semantic_loss=dict(  # Config to semantic loss
            type='CrossEntropyLoss',  # Type of loss
            reduction='sum',  # Specifies the reduction to apply to the output
            loss_weight=1.0)))  # Loss weight of the semantic loss
train_cfg = dict(  # Config of training hyperparameters for votenet
    pos_distance_thr=0.3,  # distance >= threshold 0.3 will be taken as positive samples
    neg_distance_thr=0.6,  # distance < threshold 0.6 will be taken as positive samples
    sample_mod='vote')  # Mode of the sampling method
test_cfg = dict(  # Config of testing hyperparameters for votenet
    sample_mod='seed',  # Mode of the sampling method
    nms_thr=0.25,  # The threshold to be used during NMS
    score_thr=0.8,  # Threshold to filter out boxes
    per_class_proposal=False)  # Whether to use per_class_proposal
dataset_type = 'ScanNetDataset'  # Type of the dataset
data_root = './data/scannet/'  # Root path of the data
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')  # Names of classes
train_pipeline = [  # Training pipeline, refer to mmdet3d.datasets.pipelines for more details
    dict(
        type='LoadPointsFromFile',  # First pipeline to load points, refer to mmdet3d.datasets.pipelines.indoor_loading for more details
        shift_height=True,  # Whether to use shifted height
        load_dim=6,  # The dimension of the loaded points
        use_dim=[0, 1, 2]),  # Which dimensions of the points to be used
    dict(
        type='LoadAnnotations3D',  # Second pipeline to load annotations, refer to mmdet3d.datasets.pipelines.indoor_loading for more details
        with_bbox_3d=True,  # Whether to load 3D boxes
        with_label_3d=True,  # Whether to load 3D labels
        with_mask_3d=True,  # Whether to load 3D instance masks
        with_seg_3d=True),  # Whether to load 3D semantic masks
    dict(
        type='PointSegClassMapping',  # Declare valid categories, refer to mmdet3d.datasets.pipelines.point_seg_class_mapping for more details
        valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
                       36, 39)),
    dict(type='IndoorPointSample',  # Sample indoor points, refer to mmdet3d.datasets.pipelines.indoor_sample for more details
            num_points=40000),  # Number of points to be sampled
    dict(type='IndoorFlipData',  # Augmentation pipeline that flip points and 3d boxes
        flip_ratio_yz=0.5,  # Probability of being flipped along yz plane
        flip_ratio_xz=0.5),  # Probability of being flipped along xz plane
    dict(
        type='IndoorGlobalRotScale',  # Augmentation pipeline that rotate and scale points and 3d boxes, refer to mmdet3d.datasets.pipelines.indoor_augment for more details
        shift_height=True,  # Whether to use height
        rot_range=[-0.027777777777777776, 0.027777777777777776],  # Range of rotation
        scale_range=None),  # Range of scale
    dict(
        type='DefaultFormatBundle3D',  # Default format bundle to gather data in the pipeline, refer to mmdet3d.datasets.pipelines.formating for more details
        class_names=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                     'window', 'bookshelf', 'picture', 'counter', 'desk',
                     'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                     'sink', 'bathtub', 'garbagebin')),
    dict(
        type='Collect3D',  # Pipeline that decides which keys in the data should be passed to the detector, refer to mmdet3d.datasets.pipelines.formating for more details
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask'
        ])
]
test_pipeline = [  # Testing pipeline, refer to mmdet3d.datasets.pipelines for more details
    dict(
        type='LoadPointsFromFile',  # First pipeline to load points, refer to mmdet3d.datasets.pipelines.indoor_loading for more details
        shift_height=True,  # Whether to use shifted height
        load_dim=6,  # The dimension of the loaded points
        use_dim=[0, 1, 2]),  # Which dimensions of the points to be used
    dict(type='IndoorPointSample',  # Sample indoor points, refer to mmdet3d.datasets.pipelines.indoor_sample for more details
        num_points=40000),  # Number of points to be sampled
    dict(
        type='DefaultFormatBundle3D',  # Default format bundle to gather data in the pipeline, refer to mmdet3d.datasets.pipelines.formating for more details
        class_names=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                     'window', 'bookshelf', 'picture', 'counter', 'desk',
                     'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                     'sink', 'bathtub', 'garbagebin')),
    dict(type='Collect3D',  # Pipeline that decides which keys in the data should be passed to the detector, refer to mmdet3d.datasets.pipelines.formating for more details
        keys=['points'])
]
data = dict(
    samples_per_gpu=8,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    train=dict(  # Train dataset config
        type='RepeatDataset',  # Wrapper of dataset, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/dataset_wrappers.py for details.
        times=5,  # Repeat times
        dataset=dict(
            type='ScanNetDataset',  # Type of dataset
            data_root='./data/scannet/',  # Root path of the data
            ann_file='./data/scannet/scannet_infos_train.pkl',  # Ann path of the data
            pipeline=[  # pipeline, this is passed by the train_pipeline created before.
                dict(
                    type='LoadPointsFromFile',
                    shift_height=True,
                    load_dim=6,
                    use_dim=[0, 1, 2]),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_mask_3d=True,
                    with_seg_3d=True),
                dict(
                    type='PointSegClassMapping',
                    valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24,
                                   28, 33, 34, 36, 39)),
                dict(type='IndoorPointSample', num_points=40000),
                dict(
                    type='IndoorFlipData',
                    flip_ratio_yz=0.5,
                    flip_ratio_xz=0.5),
                dict(
                    type='IndoorGlobalRotScale',
                    shift_height=True,
                    rot_range=[-0.027777777777777776, 0.027777777777777776],
                    scale_range=None),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=('cabinet', 'bed', 'chair', 'sofa', 'table',
                                 'door', 'window', 'bookshelf', 'picture',
                                 'counter', 'desk', 'curtain', 'refrigerator',
                                 'showercurtrain', 'toilet', 'sink', 'bathtub',
                                 'garbagebin')),
                dict(
                    type='Collect3D',
                    keys=[
                        'points', 'gt_bboxes_3d', 'gt_labels_3d',
                        'pts_semantic_mask', 'pts_instance_mask'
                    ])
            ],
            filter_empty_gt=False,  # Whether to filter ground empty truth boxes
            classes=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                     'window', 'bookshelf', 'picture', 'counter', 'desk',
                     'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                     'sink', 'bathtub', 'garbagebin'))),  # Names of classes
    val=dict(  # Validation dataset config
        type='ScanNetDataset',  # Type of dataset
        data_root='./data/scannet/',  # Root path of the data
        ann_file='./data/scannet/scannet_infos_val.pkl',  # Ann path of the data
        pipeline=[  # Pipeline is passed by test_pipeline created before
            dict(
                type='LoadPointsFromFile',
                shift_height=True,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='IndoorPointSample', num_points=40000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=('cabinet', 'bed', 'chair', 'sofa', 'table',
                             'door', 'window', 'bookshelf', 'picture',
                             'counter', 'desk', 'curtain', 'refrigerator',
                             'showercurtrain', 'toilet', 'sink', 'bathtub',
                             'garbagebin')),
            dict(type='Collect3D', keys=['points'])
        ],
        classes=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                 'garbagebin'),  # Names of classes
        test_mode=True),  # Whether to use test mode
    test=dict(  # Test dataset config
        type='ScanNetDataset',  # Type of dataset
        data_root='./data/scannet/',  # Root path of the data
        ann_file='./data/scannet/scannet_infos_val.pkl',  # Ann path of the data
        pipeline=[  # Pipeline is passed by test_pipeline created before
            dict(
                type='LoadPointsFromFile',
                shift_height=True,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='IndoorPointSample', num_points=40000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=('cabinet', 'bed', 'chair', 'sofa', 'table',
                             'door', 'window', 'bookshelf', 'picture',
                             'counter', 'desk', 'curtain', 'refrigerator',
                             'showercurtrain', 'toilet', 'sink', 'bathtub',
                             'garbagebin')),
            dict(type='Collect3D', keys=['points'])
        ],
        classes=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                 'garbagebin'),  # Names of classes
        test_mode=True))  # Whether to use test mode
lr = 0.008  # Learning rate of optimizers
optimizer = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
    type='Adam',  # Type of optimizers,   # Type of optimizers, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/optimizer/default_constructor.py#L13 for more details
    lr=0.008)  # Learning rate of optimizers, see detail usages of the parameters in the documentaion of PyTorch
optimizer_config = dict(  # Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
    grad_clip=dict(  # Config used to grad_clip
    max_norm=10,  # max norm of the gradients
    norm_type=2))  # Type of the used p-norm. Can be 'inf' for infinity norm.
lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',  # The policy of scheduler, also support CosineAnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    warmup=None,  # The warmup policy, also support `exp` and `constant`.
    step=[24, 32])  # Steps to decay the learning rate
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=1)  # The save interval is 1
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])  # The logger used to record the training process.
total_epochs = 36  # Total epochs to train the model
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'  # The level of logging.
find_unused_parameters = True  # Whether to find unused parameters
work_dir = None  # Directory to save the model checkpoints and logs for the current experiments.
load_from = None # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 36 epochs according to the total_epochs.
gpu_ids = range(0, 1)  # ids of gpus
```
