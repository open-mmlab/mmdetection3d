_base_ = './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'

model = dict(
    pts_backbone=dict(
        _delete_=True,  # Completely replace the base backbone config
        type='FireRPFNetV2',
        in_channels=256,  # Output channels from BEVFusionSparseEncoder
        out_channels=[128, 256, 256, 256],  # 4 stages with multi-scale outputs
        with_cbam=True,  # Enable Channel and Spatial Attention
        multi_scale_output=True,  # CRITICAL: Enable multi-scale feature extraction
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)),

    pts_neck=dict(
        _delete_=True,  # Replace the base neck config
        type='SECONDFPN',
        in_channels=[128, 256, 256, 256],  # Must match FireRPFNetV2 out_channels
        out_channels=[128, 128, 128, 128],  # Uniform output channels for fusion
        upsample_strides=[1, 1, 1, 1],  # CRITICAL: No upsampling (same resolution)
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),  # Use 1x1 conv when stride=1

    # Update bbox_head to match concatenated neck output
    # SECONDFPN concatenates all outputs: 128 * 4 = 512 channels
    bbox_head=dict(
        in_channels=512,  # 128 * 4 from SECONDFPN concatenation
    )
)

# Update the work directory to distinguish from base config
work_dir = './work_dirs/bevfusion_lidar_voxel0075_firerpfnet_with_neck_8xb4-cyclic-20e_nus-3d'
