_base_ = './pointpillars_hv_regnet-400mf_fpn_sbn-all_8xb2-2x_lyft-3d.py'
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
                [4.75, 1.92, 1.71],  # car
                [10.24, 2.84, 3.44],  # truck
                [12.70, 2.92, 3.42],  # bus
                [6.52, 2.42, 2.34],  # emergency vehicle
                [8.17, 2.75, 3.20],  # other vehicle
                [2.35, 0.96, 1.59],  # motorcycle
                [1.76, 0.63, 1.44],  # bicycle
                [0.80, 0.76, 1.76],  # pedestrian
                [0.73, 0.35, 0.50]  # animal
            ],
            rotations=[0, 1.57],
            reshape_out=True)))
