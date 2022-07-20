voxel_size = [0.32, 0.32, 0.1]

model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(90000, 120000)),

    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),

    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 384, 512],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),


    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),


    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),  
            #sub-voxel location refinement R2
            #height above ground R1
            #3D size R3
            #yaw rotation angle R2
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-60, -103.84, -3, 62.88, 60, 1],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),


    # img_roi_head=dict(
    #     type='StandardRoIHead',
    #     bbox_roi_extractor=dict(
    #         type='SingleRoIExtractor',
    #         roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
    #         out_channels=256,
    #         featmap_strides=[4, 8, 16, 32]),
    #     bbox_head=dict(
    #         type='Shared2FCBBoxHead',
    #         in_channels=256,
    #         fc_out_channels=1024,
    #         roi_feat_size=7,
    #         num_classes=80,
    #         bbox_coder=dict(
    #             type='DeltaXYWHBBoxCoder',
    #             target_means=[0., 0., 0., 0.],
    #             target_stds=[0.1, 0.1, 0.2, 0.2]),
    #         reg_class_agnostic=True,
    #         loss_cls=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #         loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    #     mask_roi_extractor=dict(
    #         type='SingleRoIExtractor',
    #         roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
    #         out_channels=256,
    #         featmap_strides=[4, 8, 16, 32]),
    #     mask_head=dict(
    #         type='FCNMaskHead',
    #         num_convs=4,
    #         in_channels=256,
    #         conv_out_channels=256,
    #         num_classes=80,
    #         loss_mask=dict(
    #             type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),

    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            # point_cloud_range=[-60, -103.84, -3, 62.88, 60, 1],
            grid_size=[384, 512, 41],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            # point_cloud_range=[-60, -103.84, -3, 62.88, 60, 1],
            post_center_limit_range=[-60, -60, -3, 60, 60, 1],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=4096,
            post_max_size=512,
            nms_thr=0.1)))





















# voxel_size = [0.32, 0.32, 4]  #pointpillar기준 voxel size
# # point_cloud_range=[-60, -103.84, -3, 62.88, 60, 1]


# model = dict(
#     type='CenterPoint',

#     pts_voxel_layer=dict(
#         max_num_points=20, 
#         voxel_size=voxel_size, 
#         max_voxels=(16000, 40000),
#         point_cloud_range=[-60, -103.84, -3, 62.88, 60, 1]),

#     pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),  ##Voxel feature encoder [voxel_num, num_per_v, num_features]->[voxel_num, num_features]
    
#     pts_middle_encoder=dict(
#         type='SparseEncoder',
#         in_channels=4,
#         sparse_shape=[1, 512, 384],
#         output_channels=128,
#         encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
#         encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
#         block_type='basicblock'), ### return [N, C * D, H, W]

#     pts_backbone=dict(
#         type='SECOND',
#         in_channels=256,
#         out_channels=[128, 256],
#         layer_nums=[5, 5],
#         layer_strides=[1, 2],
#         norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
#         conv_cfg=dict(type='Conv2d', bias=False)),

#     pts_neck=dict(
#         type='SECONDFPN',
#         in_channels=[128, 256],
#         out_channels=[256, 256],
#         upsample_strides=[1, 2],
#         norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
#         upsample_cfg=dict(type='deconv', bias=False),
#         use_conv_for_no_stride=True),

#     pts_bbox_head=dict(
#         type='CenterHead',
#         in_channels=sum([256, 256]),
#         tasks=[
#             dict(num_class=1, class_names=['Car']),
#             dict(num_class=1, class_names=['ped']),
#         ],
#         common_heads=dict(
#             reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
#         share_conv_channel=64,
#         bbox_coder=dict(
#             type='CenterPointBBoxCoder',
#             post_center_range=[-60, -103.84, -3, 62.88, 60, 1],
#             max_num=100,
#             score_threshold=0.1,
#             out_size_factor=8,
#             voxel_size=voxel_size[:2],
#             code_size=7,
# 			pc_range=[-60, -103.84] #point_cloud_range[:2],
#             ),
#         separate_head=dict(
#             type='SeparateHead', init_bias=-2.19, final_kernel=3),
#         loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
#         loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
#         norm_bbox=True),

#         train_cfg=dict(
	
#         pts=dict(
# 			point_cloud_range=[-60, -103.84, -3, 62.88, 60, 1],
#             grid_size=[384, 512, 1],
#             voxel_size=voxel_size,
#             out_size_factor=8,
#             dense_reg=1,
#             gaussian_overlap=0.1,
#             max_objs=500,
#             min_radius=2,
#             code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
#     test_cfg=dict(
#         pts=dict(
# 			point_cloud_range=[-60, -103.84, -3, 62.88, 60, 1],
#             post_center_limit_range=[-60, -103.84, -3, 62.88, 60, 1],
#             max_per_img=500,
#             max_pool_nms=False,
#             min_radius=[4, 12, 10, 1, 0.85, 0.175],
#             score_threshold=0.1,
#             out_size_factor=4,
#             voxel_size=voxel_size[:2],
#             nms_type='rotate',
#             pre_max_size=4096,
#             post_max_size=512,
#             nms_thr=0.2)))
        