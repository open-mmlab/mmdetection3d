_base_ = ['./centerpoint_0075voxel_second_secfpn_4x8_cyclic_20e_nus.py']

model = dict(test_cfg=dict(pts=dict(nms_type='circle')))
