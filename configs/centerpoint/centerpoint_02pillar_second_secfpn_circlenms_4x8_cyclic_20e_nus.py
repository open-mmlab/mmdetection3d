_base_ = ['./centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py']

test_cfg = dict(pts=dict(nms_type='circle'))
