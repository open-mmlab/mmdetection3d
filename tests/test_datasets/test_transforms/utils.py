# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

# create a dummy `results` to test the pipeline
from mmdet3d.datasets import LoadAnnotations3D, LoadPointsFromFile
from mmdet3d.datasets.transforms.loading import LoadImageFromFileMono3D
from mmdet3d.structures import LiDARInstance3DBoxes


def create_dummy_data_info(with_ann=True):

    ann_info = {
        'gt_bboxes':
        np.array([[712.4, 143., 810.73, 307.92]]),
        'gt_labels':
        np.array([1]),
        'gt_bboxes_3d':
        LiDARInstance3DBoxes(
            np.array(
                [[8.7314, -1.8559, -1.5997, 1.2000, 0.4800, 1.8900,
                  -1.5808]])),
        'gt_labels_3d':
        np.array([1]),
        'centers_2d':
        np.array([[765.04, 214.56]]),
        'depths':
        np.array([8.410]),
        'num_lidar_pts':
        np.array([377]),
        'difficulty':
        np.array([0]),
        'truncated':
        np.array([0]),
        'occluded':
        np.array([0]),
        'alpha':
        np.array([-0.2]),
        'score':
        np.array([0.]),
        'index':
        np.array([0]),
        'group_id':
        np.array([0])
    }
    data_info = {
        'sample_id':
        0,
        'images': {
            'CAM0': {
                'cam2img': [[707.0493, 0.0, 604.0814, 0.0],
                            [0.0, 707.0493, 180.5066, 0.0],
                            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            },
            'CAM1': {
                'cam2img': [[707.0493, 0.0, 604.0814, -379.7842],
                            [0.0, 707.0493, 180.5066, 0.0],
                            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            },
            'CAM2': {
                'img_path':
                'tests/data/kitti/training/image_2/000000.png',
                'height':
                370,
                'width':
                1224,
                'cam2img': [[707.0493, 0.0, 604.0814, 45.75831],
                            [0.0, 707.0493, 180.5066, -0.3454157],
                            [0.0, 0.0, 1.0, 0.004981016], [0.0, 0.0, 0.0, 1.0]]
            },
            'CAM3': {
                'cam2img': [[707.0493, 0.0, 604.0814, -334.1081],
                            [0.0, 707.0493, 180.5066, 2.33066],
                            [0.0, 0.0, 1.0, 0.003201153], [0.0, 0.0, 0.0, 1.0]]
            },
            'R0_rect': [[
                0.9999127984046936, 0.010092630051076412,
                -0.008511931635439396, 0.0
            ],
                        [
                            -0.010127290152013302, 0.9999405741691589,
                            -0.004037670791149139, 0.0
                        ],
                        [
                            0.008470674976706505, 0.0041235219687223434,
                            0.9999555945396423, 0.0
                        ], [0.0, 0.0, 0.0, 1.0]]
        },
        'lidar_points': {
            'num_pts_feats':
            4,
            'lidar_path':
            'tests/data/kitti/training/velodyne_reduced/000000.bin',
            'lidar2cam': [[
                -0.0015960992313921452, -0.9999162554740906,
                -0.012840436771512032, -0.022366708144545555
            ],
                          [
                              -0.00527064548805356, 0.012848696671426296,
                              -0.9999035596847534, -0.05967890843749046
                          ],
                          [
                              0.9999848008155823, -0.0015282672829926014,
                              -0.005290712229907513, -0.33254900574684143
                          ], [0.0, 0.0, 0.0, 1.0]],
            'Tr_velo_to_cam': [[
                0.006927963811904192, -0.9999722242355347, -0.0027578289154917,
                -0.024577289819717407
            ],
                               [
                                   -0.0011629819637164474,
                                   0.0027498360723257065, -0.9999955296516418,
                                   -0.06127237156033516
                               ],
                               [
                                   0.999975323677063, 0.006931141018867493,
                                   -0.0011438990477472544, -0.33210289478302
                               ], [0.0, 0.0, 0.0, 1.0]],
            'Tr_imu_to_velo': [[
                0.999997615814209, 0.0007553070900030434,
                -0.002035825978964567, -0.8086758852005005
            ],
                               [
                                   -0.0007854027207940817, 0.9998897910118103,
                                   -0.014822980388998985, 0.3195559084415436
                               ],
                               [
                                   0.002024406101554632, 0.014824540354311466,
                                   0.9998881220817566, -0.7997230887413025
                               ], [0.0, 0.0, 0.0, 1.0]]
        },
        'instances': [{
            'bbox': [712.4, 143.0, 810.73, 307.92],
            'bbox_label':
            -1,
            'bbox_3d': [
                1.840000033378601, 1.4700000286102295, 8.40999984741211,
                1.2000000476837158, 1.8899999856948853, 0.47999998927116394,
                0.009999999776482582
            ],
            'bbox_label_3d':
            -1,
            'center_2d': [765.04, 214.56],
            'depth':
            8.410,
            'num_lidar_pts':
            377,
            'difficulty':
            0,
            'truncated':
            0,
            'occluded':
            0,
            'alpha':
            -0.2,
            'score':
            0.0,
            'index':
            0,
            'group_id':
            0
        }],
        'plane':
        None
    }
    if with_ann:
        data_info['ann_info'] = ann_info
    return data_info


def create_data_info_after_loading():
    load_anns_transform = LoadAnnotations3D(
        with_bbox_3d=True, with_label_3d=True)
    load_points_transform = LoadPointsFromFile(
        coord_type='LIDAR', load_dim=4, use_dim=3)
    data_info = create_dummy_data_info()
    data_info = load_points_transform(data_info)
    data_info_after_loading = load_anns_transform(data_info)
    return data_info_after_loading


def create_mono3d_data_info_after_loading():
    load_anns_transform = LoadAnnotations3D(
        with_bbox=True,
        with_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True)
    load_img_transform = LoadImageFromFileMono3D()
    data_info = create_dummy_data_info()
    data_info = load_img_transform(data_info)
    data_info_after_loading = load_anns_transform(data_info)
    return data_info_after_loading
