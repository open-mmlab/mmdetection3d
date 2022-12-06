_base_ = './dgcnn_4xb32-cosine-100e_s3dis-seg_test-area5.py'

# data settings
train_area = [1, 2, 4, 5, 6]
test_area = 3
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_files=[f's3dis_infos_Area_{i}.pkl' for i in train_area],
        scene_idxs=[
            f'seg_info/Area_{i}_resampled_scene_idxs.npy' for i in train_area
        ]))
test_dataloader = dict(
    dataset=dict(
        ann_files=f's3dis_infos_Area_{test_area}.pkl',
        scene_idxs=f'seg_info/Area_{test_area}_resampled_scene_idxs.npy'))
val_dataloader = test_dataloader
