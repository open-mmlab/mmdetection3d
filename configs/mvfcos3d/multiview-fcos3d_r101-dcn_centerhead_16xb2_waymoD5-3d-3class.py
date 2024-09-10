_base_ = ['./multiview-dfm_r101-dcn_16xb2_waymoD5-3d-3class.py']

model = dict(
    bbox_head=dict(
        _delete_=True,
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['Car']),
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-35.0, -75.0, -2, 75.0, 75.0, 4],
            pc_range=[-35.0, -75.0, -2, 75.0, 75.0, 4],
            max_num=2000,
            score_threshold=0,
            out_size_factor=1,
            voxel_size=(.50, .50),
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        _delete_=True,
        grid_size=[220, 300, 1],
        voxel_size=(0.5, 0.5, 6),
        out_size_factor=1,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        point_cloud_range=[-35.0, -75.0, -2, 75.0, 75.0, 4]),
    test_cfg=dict(
        _delete_=True,
        post_center_limit_range=[-35.0, -75.0, -2, 75.0, 75.0, 4],
        max_per_img=4096,
        max_pool_nms=False,
        min_radius=[0.5, 2, 6],
        score_threshold=0,
        out_size_factor=1,
        voxel_size=(0.5, 0.5),
        nms_type='circle',
        pre_max_size=2000,
        post_max_size=200,
        nms_thr=0.2))
