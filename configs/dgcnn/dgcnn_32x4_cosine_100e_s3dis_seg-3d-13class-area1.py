_base_ = [
    '../_base_/datasets/s3dis_seg-3d-13class.py', '../_base_/models/dgcnn.py',
    '../_base_/schedules/seg_cosine_100e.py', '../_base_/default_runtime.py'
]

# data settings
train_area = [2, 3, 4, 5, 6]
test_area = 1
data_root = './data/s3dis/'
data = dict(
    samples_per_gpu=32,
    train=dict(
        ann_files=[
            data_root + f's3dis_infos_Area_{i}.pkl' for i in train_area
        ],
        scene_idxs=[
            data_root + f'seg_info/Area_{i}_resampled_scene_idxs.npy'
            for i in train_area
        ]),
    val=dict(
        ann_files=data_root + f's3dis_infos_Area_{test_area}.pkl',
        scene_idxs=data_root +
        f'seg_info/Area_{test_area}_resampled_scene_idxs.npy'),
    test=dict(ann_files=data_root + f's3dis_infos_Area_{test_area}.pkl'))

evaluation = dict(interval=2)

# model settings
model = dict(
    backbone=dict(in_channels=9),  # [xyz, rgb, normalized_xyz]
    decode_head=dict(
        num_classes=13, ignore_index=13,
        loss_decode=dict(class_weight=None)),  # S3DIS doesn't use class_weight
    test_cfg=dict(
        num_points=4096,
        block_size=1.0,
        sample_rate=0.5,
        use_normalized_coord=True,
        batch_size=24))

# runtime settings
checkpoint_config = dict(interval=2)
