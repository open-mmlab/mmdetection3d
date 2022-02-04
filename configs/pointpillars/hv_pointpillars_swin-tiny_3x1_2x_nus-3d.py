_base_ = [
    '../_base_/models/hv_pointpillars_fpn_nus.py',
    '../_base_/datasets/nus-3d.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
voxel_size = [0.25, 0.25, 8]
model = dict(
pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
pts_voxel_encoder=dict(
        point_cloud_range=[-50, -50, -5, 50, 50, 3]),
pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[400, 400]),
pts_backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=400,
        patch_size = 2,
        in_channels=64,
        embed_dims=96,
        strides = [2, 2, 2],
        depths=[2, 2, 6],
        num_heads=[3, 6, 12],
        window_size=20,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2),
        with_cp=False,
        convert_weights=True,
        ),
pts_neck=dict(
        in_channels=[96, 192, 384],
        out_channels=256,
        num_outs=3)
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3
)