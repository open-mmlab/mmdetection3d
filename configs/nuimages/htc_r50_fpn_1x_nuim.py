_base_ = './htc_r50_fpn_head-without-semantic_1x_nuim.py'
model = dict(
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=32,
            ignore_label=0,
            loss_weight=0.2)))

data_root = 'data/nuimages/'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(
        type='Resize',
        img_scale=[(1280, 720), (1920, 1080)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='PackDetInputs')
]
data = dict(
    train=dict(
        seg_prefix=data_root + 'annotations/semantic_masks/',
        pipeline=train_pipeline))
