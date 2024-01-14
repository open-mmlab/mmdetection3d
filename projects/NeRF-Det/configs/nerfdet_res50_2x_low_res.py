_base_ = ['./nerfdet_res50_2x_low_res_depth.py']

model = dict(depth_supervise=False)

dataset_type = 'MultiViewScanNetDataset'
data_root = 'data/scannet/'
class_names = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
    'toilet', 'sink', 'bathtub', 'garbagebin'
]
metainfo = dict(CLASSES=class_names)
file_client_args = dict(backend='disk')

input_modality = dict(use_depth=False)
backend_args = None

train_collect_keys = [
    'img', 'gt_bboxes_3d', 'gt_labels_3d', 'lightpos', 'nerf_sizes', 'raydirs',
    'gt_images', 'gt_depths', 'denorm_images'
]

test_collect_keys = [
    'img',
    'lightpos',
    'nerf_sizes',
    'raydirs',
    'gt_images',
    'gt_depths',
    'denorm_images',
]

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=50,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='Resize', scale=(320, 240), keep_ratio=True),
        ],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        margin=10,
        depth_range=[0.5, 5.5],
        loading='random',
        nerf_target_views=10),
    dict(type='RandomShiftOrigin', std=(.7, .7, .0)),
    dict(type='PackNeRFDetInputs', keys=train_collect_keys)
]

test_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=101,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='Resize', scale=(320, 240), keep_ratio=True),
        ],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        margin=10,
        depth_range=[0.5, 5.5],
        loading='random',
        nerf_target_views=1),
    dict(type='PackNeRFDetInputs', keys=test_collect_keys)
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='scannet_infos_train_new.pkl',
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            filter_empty_gt=True,
            box_type_3d='Depth',
            metainfo=metainfo)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_infos_val_new.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        filter_empty_gt=True,
        box_type_3d='Depth',
        metainfo=metainfo))
test_dataloader = val_dataloader
