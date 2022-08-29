_base_ = [
    '../_base_/datasets/sunrgbd-3d.py', '../_base_/default_runtime.py',
    '../_base_/models/imvotenet.py'
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_bbox_3d=False,
        with_label_3d=False),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 480), (1333, 504), (1333, 528), (1333, 552),
                (1333, 576), (1333, 600)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pack3DDetInputs', keys=['img', 'gt_bboxes', 'gt_bboxes_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # online evaluation
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_bbox_3d=False,
        with_label_3d=False),
    dict(type='Resize', scale=(1333, 600), keep_ratio=True),
    dict(
        type='Pack3DDetInputs',
        keys=(['img']),
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset', times=1, dataset=dict(pipeline=train_pipeline)))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        milestones=[6],
        gamma=0.1)
]
val_evaluator = dict(type='Indoor2DMetric')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'  # noqa
