_base_ = ['./centerpoint_voxel01-second-secfpn_8xb4-cyclic-20e_nus.py']

model = dict(test_cfg=dict(pts=dict(nms_type='circle')))
