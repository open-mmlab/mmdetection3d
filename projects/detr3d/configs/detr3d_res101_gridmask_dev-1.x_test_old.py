_base_ = [
    './detr3d_res101_gridmask_dev-1.x.py'
]
model = dict(type='Detr3D_old')
# mAP: 0.3469
# mATE: 0.7651
# mASE: 0.2678
# mAOE: 0.3916
# mAVE: 0.8758
# mAAE: 0.2110
# NDS: 0.4223