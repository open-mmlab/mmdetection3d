_base_ = './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'

# Override the point cloud backbone with FireRPFNet
# FireRPFNet is more memory-efficient than SECOND while maintaining good performance
# It uses Fire modules (SqueezeNet-style) with CBAM attention
model = dict(
    pts_backbone=dict(
        _delete_=True,  # Completely replace the base backbone config
        type='FireRPFNetV2',
        in_channels=256,  # Output channels from BEVFusionSparseEncoder
        out_channels=[128, 256, 256, 512],  # 4 stages with increasing channels
        with_cbam=True,  # Enable Channel and Spatial Attention
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)),
    pts_neck=None
)

# Update the work directory
work_dir = './work_dirs/bevfusion_lidar_voxel0075_firerpfnet_8xb4-cyclic-20e_nus-3d'
