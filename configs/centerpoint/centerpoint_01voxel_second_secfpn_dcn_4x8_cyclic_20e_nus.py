_base_ = ['./centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py']

model = dict(pts_bbox_head=dict(dcn_head=True))
