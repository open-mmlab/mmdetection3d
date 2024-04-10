# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['./cenet_64x2048_8xb2_50e_semantickitti.py']

backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='LaserMix',
                    num_areas=[3, 4, 5, 6],
                    pitch_angles=[-25, 3],
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4),
                        dict(
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.int32',
                            seg_offset=2**16,
                            dataset_type='semantickitti'),
                        dict(type='PointSegClassMapping')
                    ],
                    prob=1)
            ],
            [
                dict(
                    type='PolarMix',
                    instance_classes=[0, 1, 2, 3, 4, 5, 6, 7],
                    swap_ratio=0.5,
                    rotate_paste_ratio=1.0,
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4),
                        dict(
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.int32',
                            seg_offset=2**16,
                            dataset_type='semantickitti'),
                        dict(type='PointSegClassMapping')
                    ],
                    prob=1)
            ],
        ],
        prob=[0.5, 0.5]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(
        type='SemkittiRangeView',
        H=64,
        W=2048,
        fov_up=3.0,
        fov_down=-25.0,
        means=(11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
        stds=(10.24, 12.295865, 9.4287, 0.8643, 0.1450),
        ignore_index=19),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_semantic_seg'])
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
