_base_ = ['./spvcnn_8xb2-15e_semtickitti.py']

# file_client_args = dict(backend='disk')
# train_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         file_client_args=file_client_args),
#     dict(
#         type='LoadAnnotations3D',
#         with_bbox_3d=False,
#         with_label_3d=False,
#         with_seg_3d=True,
#         seg_3d_dtype='np.int32',
#         seg_offset=2**16,
#         dataset_type='semantickitti'),
#     dict(type='PointSegClassMapping'),
#     dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
# ]

# train_dataloader = dict(
#     num_workers=0,
#     dataset=dict(dataset=dict(
#             pipeline=train_pipeline))

# )
model = dict(
    backbone=dict(
        base_channels=16,
        enc_channels=[16, 32, 64, 128],
        dec_channels=[128, 64, 48, 48]),
    decode_head=dict(channels=48))

# load_from='checkpoints/spvcnn_init.pth'
