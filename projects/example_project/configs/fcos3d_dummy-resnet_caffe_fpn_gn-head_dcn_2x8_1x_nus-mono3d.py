_base_ = [
    '../../../configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py'  # noqa
]

custom_imports = dict(imports=['projects.example_project.dummy'])

_base_.model.backbone.type = 'DummyResNet'
