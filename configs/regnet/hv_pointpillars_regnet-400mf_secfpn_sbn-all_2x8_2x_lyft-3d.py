_base_ = './hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d.py'
# model settings
model = dict(
    pts_neck=dict(
        type='SECONDFPN',
        _delete_=True,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 160, 384],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        in_channels=384,
        feat_channels=384,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-80, -80, -1.0715024, 80, 80, -1.0715024],
                    [-80, -80, -0.3033737, 80, 80, -0.3033737],
                    [-80, -80, -0.3519405, 80, 80, -0.3519405],
                    [-80, -80, -0.8871424, 80, 80, -0.8871424],
                    [-80, -80, -0.6276341, 80, 80, -0.6276341],
                    [-80, -80, -1.3220503, 80, 80, -1.3220503],
                    [-80, -80, -1.0709302, 80, 80, -1.0709302],
                    [-80, -80, -0.9122268, 80, 80, -0.9122268],
                    [-80, -80, -1.8012227, 80, 80, -1.8012227]],
            sizes=[
                [1.92, 4.75, 1.71],  # car
                [2.84, 10.24, 3.44],  # truck
                [2.92, 12.70, 3.42],  # bus
                [2.42, 6.52, 2.34],  # emergency vehicle
                [2.75, 8.17, 3.20],  # other vehicle
                [0.96, 2.35, 1.59],  # motorcycle
                [0.63, 1.76, 1.44],  # bicycle
                [0.76, 0.80, 1.76],  # pedestrian
                [0.35, 0.73, 0.50]  # animal
            ],
            rotations=[0, 1.57],
            reshape_out=True)))
