_base_ = ['./centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py']

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(nms_type='circle')))
