_base_ = [
    '../_base_/models/pointpillars_hv_fpn_range100_lyft.py',
    '../_base_/datasets/lyft-3d-range100.py',
    '../_base_/schedules/schedule-2x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        in_channels=384,
        feat_channels=384,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-100, -100, -1.0715024, 100, 100, -1.0715024],
                    [-100, -100, -0.3033737, 100, 100, -0.3033737],
                    [-100, -100, -0.3519405, 100, 100, -0.3519405],
                    [-100, -100, -0.8871424, 100, 100, -0.8871424],
                    [-100, -100, -0.6276341, 100, 100, -0.6276341],
                    [-100, -100, -1.3220503, 100, 100, -1.3220503],
                    [-100, -100, -1.0709302, 100, 100, -1.0709302],
                    [-100, -100, -0.9122268, 100, 100, -0.9122268],
                    [-100, -100, -1.8012227, 100, 100, -1.8012227]],
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
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
