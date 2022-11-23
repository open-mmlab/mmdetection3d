_base_ = [
    '../../../configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py'  # noqa
]

custom_imports = dict(imports=['projects.example_project.dummy'])

_base_.model.backbone.type = 'DummyResNet'
