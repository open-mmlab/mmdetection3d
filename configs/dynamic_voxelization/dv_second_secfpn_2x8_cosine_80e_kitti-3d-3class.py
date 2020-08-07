_base_ = '../second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py'

point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size = [0.05, 0.05, 0.1]

model = dict(
    type='DynamicVoxelNet',
    voxel_layer=dict(
        _delete_=True,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(-1, -1)),
    voxel_encoder=dict(
        _delete_=True,
        type='DynamicSimpleVFE',
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range))

# optimizer
lr = 0.003  # max learning rate
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99),  # the momentum is change during training
    weight_decay=0.001)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
momentum_config = None
