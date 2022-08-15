# Copyright (c) OpenMMLab. All rights reserved.

# missing regnet/sassd/nuimages

# yapf: disable
ssd3d = dict(
    config='configs/3dssd/3dssd_4x4_kitti-3d-car.py',
    checkpoint='3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth',
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/3dssd/3dssd_4x4_kitti-3d-car/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=78.58),
)
centerpoint = dict(
    config='configs/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py', # noqa
    checkpoint='centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210816_064624-0f3299c0.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210816_064624-0f3299c0.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=59.66),
)
dgcnn = dict(
    config='configs/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py',
    checkpoint='dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210731_000734-39658f14.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection3d/v0.17.0_models/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class/area1/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class_20210731_000734-39658f14.pth', # noqa
    metric=dict(mIOU=68.33),
)
dynamic_voxelization = dict(
        config='configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py', # noqa
        checkpoint='dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class_20210831_054106-e742d163.pth', # noqa
        eval='bbox',
        url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class_20210831_054106-e742d163.pth', # noqa
        metric=dict(bbox_mAP=65.27),
)
fcos_3d = dict(
    config='configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py', # noqa
    checkpoint='fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=29.8),
)
free_anchor = dict(
    config='configs/centernet/centernet_resnet18_dcnv2_140e_coco.py',
    checkpoint='hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d_20210828_025608-bfbd506e.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d_20210828_025608-bfbd506e.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=52.04),
)
groupfree3d = dict(
    config='configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256.py',  # noqa
    checkpoint='groupfree3d_8x4_scannet-3d-18class-L6-O256_20210702_145347-3499eb55.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection3d/v0.1.0_models/groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256/groupfree3d_8x4_scannet-3d-18class-L6-O256_20210702_145347-3499eb55.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=47.82),
)
h3dnet = dict(
    config='configs/h3dnet/h3dnet_3x8_scannet-3d-18class.py', # noqa
    checkpoint='h3dnet_3x8_scannet-3d-18class_20210824_003149-414bd304.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/h3dnet/h3dnet_scannet-3d-18class/h3dnet_3x8_scannet-3d-18class_20210824_003149-414bd304.pth', # noqa
    metric=dict(bbox_mAP=47.68),
)
imvotenet = dict(
    config='configs/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py', # noqa
    checkpoint='imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=62.70),
)
imvoxelnet = dict(
    config='configs/imvoxelnet/imvoxelnet_4x8_kitti-3d-car.py',
    checkpoint='imvoxelnet_4x8_kitti-3d-car_20210830_003014-3d0ffdf4.pth',
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/imvoxelnet/imvoxelnet_4x8_kitti-3d-car/imvoxelnet_4x8_kitti-3d-car_20210830_003014-3d0ffdf4.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=17.26),
)
mvxnet = dict(
    config='configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py', # noqa
    checkpoint='dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=63.22),
)
paconv = dict(
    config='configs/paconv/paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class.py',
    checkpoint='paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class_20210729_200615-2147b2d1.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v0.1.0_models/paconv/paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class/paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class_20210729_200615-2147b2d1.pth', # noqa
    metric=dict(mIOU=66.65),
)
parta2 = dict(
    config='configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py',
    checkpoint='hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20210831_022017-454a5344.pth',  # noqa
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20210831_022017-454a5344.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=68.33),
)
pgd = dict(
    config='configs/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d.py',
    checkpoint='pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth', # noqa
    eval=['bbox'],
    metric=dict(bbox_mAP=18.33),
)
point_rcnn = dict(
    config='configs/point_rcnn/point_rcnn_2x8_kitti-3d-3classes.py',
    checkpoint='point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth',
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/point_rcnn/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=70.83),
)
pointnet2 = dict(
    config='configs/pointnet2/pointnet2_ssg_xyz-only_16x2_cosine_200e_scannet_seg-3d-20class.py', # noqa
    checkpoint='pointnet2_ssg_xyz-only_16x2_cosine_200e_scannet_seg-3d-20class_20210514_143628-4e341a48.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointnet2/pointnet2_ssg_xyz-only_16x2_cosine_200e_scannet_seg-3d-20class/pointnet2_ssg_xyz-only_16x2_cosine_200e_scannet_seg-3d-20class_20210514_143628-4e341a48.pth', # noqa
    metric=dict(mIOU=53.91),
)
pointpillars = dict(
    config='configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py', # noqa
    checkpoint='hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=77.6),
)
second = dict(
    config='configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py',
    checkpoint='hv_second_secfpn_6x8_80e_kitti-3d-3class_20210831_022017-ae782e87.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-3class/hv_second_secfpn_6x8_80e_kitti-3d-3class_20210831_022017-ae782e87.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=65.74),
)
smoke = dict(
    config='configs/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py', # noqa
    checkpoint='smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v0.1.0_models/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=13.85),
)
ssn = dict(
    config='configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d.py',
    checkpoint='hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d_20210830_101351-51915986.pth', # noqa
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d_20210830_101351-51915986.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=54.44),
)
votenet = dict(
    config='configs/votenet/votenet_8x8_scannet-3d-18class.py',
    checkpoint='votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth',
    url='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/votenet/votenet_8x8_scannet-3d-18class/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth', # noqa
    eval='bbox',
    metric=dict(bbox_mAP=40.82),
)
# yapf: enable
