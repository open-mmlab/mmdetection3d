_base_ = ['./centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

model = dict(test_cfg=dict(pts=dict(nms_type='circle')))
