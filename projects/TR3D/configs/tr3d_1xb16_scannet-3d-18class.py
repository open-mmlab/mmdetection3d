_base_ = [
    '../../../configs/fcaf3d/fcaf3d_2xb8_scannet-3d-18class.py'
]

custom_imports = dict(imports=['projects.TR3D.tr3d'])

model = dict(
    backbone=dict(type='TR3DMinkResNet'),
    bbox_head=dict(type='TR3DHead'))
