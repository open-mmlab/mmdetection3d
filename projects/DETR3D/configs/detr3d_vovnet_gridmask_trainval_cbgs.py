_base_ = ['./detr3d_r101_gridmask_cbgs.py']

custom_imports = dict(imports=['projects.DETR3D.detr3d'])

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    bgr_to_rgb=False)

# this means type='DETR3D' will be processed as 'mmdet3d.DETR3D'
default_scope = 'mmdet3d'
model = dict(
    type='DETR3D',
    use_grid_mask=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor', **img_norm_cfg, pad_size_divisor=32),
    img_backbone=dict(
        _delete_=True,
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=1,
        input_ch=3,
        out_features=['stage2', 'stage3', 'stage4', 'stage5']),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True))

train_dataloader = dict(
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(ann_file='nuscenes_infos_trainval.pkl')))

test_dataloader = dict(
    dataset=dict(
        data_root='data/nuscenes-test', ann_file='nuscenes_infos_test.pkl'))

test_evaluator = dict(
    type='NuScenesMetric',
    data_root='data/nuscenes-test',
    ann_file='data/nuscenes-test/nuscenes_infos_test.pkl',
    jsonfile_prefix='work_dirs/detr3d_vovnet_results_test',
    format_only=True,
    metric=[])

load_from = 'ckpts/dd3d_det_final.pth'
find_unused_parameters = True
