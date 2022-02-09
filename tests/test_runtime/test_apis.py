# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from os.path import dirname, exists, join

import numpy as np
import pytest
import torch
from mmcv.parallel import MMDataParallel

from mmdet3d.apis import (convert_SyncBN, inference_detector,
                          inference_mono_3d_detector,
                          inference_multi_modality_detector,
                          inference_segmentor, init_model, show_result_meshlab,
                          single_gpu_test)
from mmdet3d.core import Box3DMode
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes)
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection3d repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet3d
        repo_dpath = dirname(dirname(mmdet3d.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def test_convert_SyncBN():
    cfg = _get_config_module(
        'pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py')
    model_cfg = cfg.model
    convert_SyncBN(model_cfg)
    assert model_cfg['pts_voxel_encoder']['norm_cfg']['type'] == 'BN1d'
    assert model_cfg['pts_backbone']['norm_cfg']['type'] == 'BN2d'
    assert model_cfg['pts_neck']['norm_cfg']['type'] == 'BN2d'


def test_show_result_meshlab():
    pcd = 'tests/data/nuscenes/samples/LIDAR_TOP/n015-2018-08-02-17-16-37+' \
              '0800__LIDAR_TOP__1533201470948018.pcd.bin'
    box_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[8.7314, -1.8559, -1.5997, 0.4800, 1.2000, 1.8900, 0.0100]]))
    labels_3d = torch.tensor([0])
    scores_3d = torch.tensor([0.5])
    points = np.random.rand(100, 4)
    img_meta = dict(
        pts_filename=pcd, boxes_3d=box_3d, box_mode_3d=Box3DMode.LIDAR)
    data = dict(points=[[torch.tensor(points)]], img_metas=[[img_meta]])
    result = [
        dict(
            pts_bbox=dict(
                boxes_3d=box_3d, labels_3d=labels_3d, scores_3d=scores_3d))
    ]
    tmp_dir = tempfile.TemporaryDirectory()
    temp_out_dir = tmp_dir.name
    out_dir, file_name = show_result_meshlab(data, result, temp_out_dir)
    expected_outfile_pred = file_name + '_pred.obj'
    expected_outfile_pts = file_name + '_points.obj'
    expected_outfile_pred_path = os.path.join(out_dir, file_name,
                                              expected_outfile_pred)
    expected_outfile_pts_path = os.path.join(out_dir, file_name,
                                             expected_outfile_pts)
    assert os.path.exists(expected_outfile_pred_path)
    assert os.path.exists(expected_outfile_pts_path)
    tmp_dir.cleanup()

    # test multi-modality show
    # indoor scene
    pcd = 'tests/data/sunrgbd/points/000001.bin'
    filename = 'tests/data/sunrgbd/sunrgbd_trainval/image/000001.jpg'
    box_3d = DepthInstance3DBoxes(
        torch.tensor(
            [[-1.1580, 3.3041, -0.9961, 0.3829, 0.4647, 0.5574, 1.1213]]))
    img = np.random.randn(1, 3, 608, 832)
    k_mat = np.array([[529.5000, 0.0000, 365.0000],
                      [0.0000, 529.5000, 265.0000], [0.0000, 0.0000, 1.0000]])
    rt_mat = np.array([[0.9980, 0.0058, -0.0634], [0.0058, 0.9835, 0.1808],
                       [0.0634, -0.1808, 0.9815]])
    rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ rt_mat.transpose(
        1, 0)
    depth2img = k_mat @ rt_mat
    img_meta = dict(
        filename=filename,
        depth2img=depth2img,
        pcd_horizontal_flip=False,
        pcd_vertical_flip=False,
        box_mode_3d=Box3DMode.DEPTH,
        box_type_3d=DepthInstance3DBoxes,
        pcd_trans=np.array([0., 0., 0.]),
        pcd_scale_factor=1.0,
        pts_filename=pcd,
        transformation_3d_flow=['R', 'S', 'T'])
    data = dict(
        points=[[torch.tensor(points)]], img_metas=[[img_meta]], img=[img])
    result = [dict(boxes_3d=box_3d, labels_3d=labels_3d, scores_3d=scores_3d)]
    tmp_dir = tempfile.TemporaryDirectory()
    temp_out_dir = tmp_dir.name
    out_dir, file_name = show_result_meshlab(
        data, result, temp_out_dir, 0.3, task='multi_modality-det')
    expected_outfile_pred = file_name + '_pred.obj'
    expected_outfile_pts = file_name + '_points.obj'
    expected_outfile_png = file_name + '_img.png'
    expected_outfile_proj = file_name + '_pred.png'
    expected_outfile_pred_path = os.path.join(out_dir, file_name,
                                              expected_outfile_pred)
    expected_outfile_pts_path = os.path.join(out_dir, file_name,
                                             expected_outfile_pts)
    expected_outfile_png_path = os.path.join(out_dir, file_name,
                                             expected_outfile_png)
    expected_outfile_proj_path = os.path.join(out_dir, file_name,
                                              expected_outfile_proj)
    assert os.path.exists(expected_outfile_pred_path)
    assert os.path.exists(expected_outfile_pts_path)
    assert os.path.exists(expected_outfile_png_path)
    assert os.path.exists(expected_outfile_proj_path)
    tmp_dir.cleanup()
    # outdoor scene
    pcd = 'tests/data/kitti/training/velodyne_reduced/000000.bin'
    filename = 'tests/data/kitti/training/image_2/000000.png'
    box_3d = LiDARInstance3DBoxes(
        torch.tensor(
            [[6.4495, -3.9097, -1.7409, 1.5063, 3.1819, 1.4716, 1.8782]]))
    img = np.random.randn(1, 3, 384, 1280)
    lidar2img = np.array(
        [[6.09695435e+02, -7.21421631e+02, -1.25125790e+00, -1.23041824e+02],
         [1.80384201e+02, 7.64479828e+00, -7.19651550e+02, -1.01016693e+02],
         [9.99945343e-01, 1.24365499e-04, 1.04513029e-02, -2.69386917e-01],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    img_meta = dict(
        filename=filename,
        pcd_horizontal_flip=False,
        pcd_vertical_flip=False,
        box_mode_3d=Box3DMode.LIDAR,
        box_type_3d=LiDARInstance3DBoxes,
        pcd_trans=np.array([0., 0., 0.]),
        pcd_scale_factor=1.0,
        pts_filename=pcd,
        lidar2img=lidar2img)
    data = dict(
        points=[[torch.tensor(points)]], img_metas=[[img_meta]], img=[img])
    result = [
        dict(
            pts_bbox=dict(
                boxes_3d=box_3d, labels_3d=labels_3d, scores_3d=scores_3d))
    ]
    out_dir, file_name = show_result_meshlab(
        data, result, temp_out_dir, 0.1, task='multi_modality-det')
    tmp_dir = tempfile.TemporaryDirectory()
    temp_out_dir = tmp_dir.name
    expected_outfile_pred = file_name + '_pred.obj'
    expected_outfile_pts = file_name + '_points.obj'
    expected_outfile_png = file_name + '_img.png'
    expected_outfile_proj = file_name + '_pred.png'
    expected_outfile_pred_path = os.path.join(out_dir, file_name,
                                              expected_outfile_pred)
    expected_outfile_pts_path = os.path.join(out_dir, file_name,
                                             expected_outfile_pts)
    expected_outfile_png_path = os.path.join(out_dir, file_name,
                                             expected_outfile_png)
    expected_outfile_proj_path = os.path.join(out_dir, file_name,
                                              expected_outfile_proj)
    assert os.path.exists(expected_outfile_pred_path)
    assert os.path.exists(expected_outfile_pts_path)
    assert os.path.exists(expected_outfile_png_path)
    assert os.path.exists(expected_outfile_proj_path)
    tmp_dir.cleanup()
    # test mono-3d show
    filename = 'tests/data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-' \
               '07-18-11-07-57+0800__CAM_BACK_LEFT__1531883530447423.jpg'
    box_3d = CameraInstance3DBoxes(
        torch.tensor(
            [[6.4495, -3.9097, -1.7409, 1.5063, 3.1819, 1.4716, 1.8782]]))
    img = np.random.randn(1, 3, 384, 1280)
    cam2img = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0],
                        [0.0, 0.0, 1.0]])
    img_meta = dict(
        filename=filename,
        pcd_horizontal_flip=False,
        pcd_vertical_flip=False,
        box_mode_3d=Box3DMode.CAM,
        box_type_3d=CameraInstance3DBoxes,
        pcd_trans=np.array([0., 0., 0.]),
        pcd_scale_factor=1.0,
        cam2img=cam2img)
    data = dict(
        points=[[torch.tensor(points)]], img_metas=[[img_meta]], img=[img])
    result = [
        dict(
            img_bbox=dict(
                boxes_3d=box_3d, labels_3d=labels_3d, scores_3d=scores_3d))
    ]
    out_dir, file_name = show_result_meshlab(
        data, result, temp_out_dir, 0.1, task='mono-det')
    tmp_dir = tempfile.TemporaryDirectory()
    temp_out_dir = tmp_dir.name
    expected_outfile_png = file_name + '_img.png'
    expected_outfile_proj = file_name + '_pred.png'
    expected_outfile_png_path = os.path.join(out_dir, file_name,
                                             expected_outfile_png)
    expected_outfile_proj_path = os.path.join(out_dir, file_name,
                                              expected_outfile_proj)
    assert os.path.exists(expected_outfile_png_path)
    assert os.path.exists(expected_outfile_proj_path)
    tmp_dir.cleanup()

    # test seg show
    pcd = 'tests/data/scannet/points/scene0000_00.bin'
    points = np.random.rand(100, 6)
    img_meta = dict(pts_filename=pcd)
    data = dict(points=[[torch.tensor(points)]], img_metas=[[img_meta]])
    pred_seg = torch.randint(0, 20, (100, ))
    result = [dict(semantic_mask=pred_seg)]
    tmp_dir = tempfile.TemporaryDirectory()
    temp_out_dir = tmp_dir.name
    out_dir, file_name = show_result_meshlab(
        data, result, temp_out_dir, task='seg')
    expected_outfile_pred = file_name + '_pred.obj'
    expected_outfile_pts = file_name + '_points.obj'
    expected_outfile_pred_path = os.path.join(out_dir, file_name,
                                              expected_outfile_pred)
    expected_outfile_pts_path = os.path.join(out_dir, file_name,
                                             expected_outfile_pts)
    assert os.path.exists(expected_outfile_pred_path)
    assert os.path.exists(expected_outfile_pts_path)
    tmp_dir.cleanup()


def test_inference_detector():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    pcd = 'tests/data/kitti/training/velodyne_reduced/000000.bin'
    detector_cfg = 'configs/pointpillars/hv_pointpillars_secfpn_' \
                   '6x8_160e_kitti-3d-3class.py'
    detector = init_model(detector_cfg, device='cuda:0')
    results = inference_detector(detector, pcd)
    bboxes_3d = results[0][0]['boxes_3d']
    scores_3d = results[0][0]['scores_3d']
    labels_3d = results[0][0]['labels_3d']
    assert bboxes_3d.tensor.shape[0] >= 0
    assert bboxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0


def test_inference_multi_modality_detector():
    # these two multi-modality models both only have GPU implementations
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    # indoor scene
    pcd = 'tests/data/sunrgbd/points/000001.bin'
    img = 'tests/data/sunrgbd/sunrgbd_trainval/image/000001.jpg'
    ann_file = 'tests/data/sunrgbd/sunrgbd_infos.pkl'
    detector_cfg = 'configs/imvotenet/imvotenet_stage2_'\
                   '16x8_sunrgbd-3d-10class.py'
    detector = init_model(detector_cfg, device='cuda:0')
    results = inference_multi_modality_detector(detector, pcd, img, ann_file)
    bboxes_3d = results[0][0]['boxes_3d']
    scores_3d = results[0][0]['scores_3d']
    labels_3d = results[0][0]['labels_3d']
    assert bboxes_3d.tensor.shape[0] >= 0
    assert bboxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0

    # outdoor scene
    pcd = 'tests/data/kitti/training/velodyne_reduced/000000.bin'
    img = 'tests/data/kitti/training/image_2/000000.png'
    ann_file = 'tests/data/kitti/kitti_infos_train.pkl'
    detector_cfg = 'configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_' \
                   '2x8_80e_kitti-3d-3class.py'
    detector = init_model(detector_cfg, device='cuda:0')
    results = inference_multi_modality_detector(detector, pcd, img, ann_file)
    bboxes_3d = results[0][0]['pts_bbox']['boxes_3d']
    scores_3d = results[0][0]['pts_bbox']['scores_3d']
    labels_3d = results[0][0]['pts_bbox']['labels_3d']
    assert bboxes_3d.tensor.shape[0] >= 0
    assert bboxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0


def test_inference_mono_3d_detector():
    # FCOS3D only has GPU implementations
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    img = 'tests/data/nuscenes/samples/CAM_BACK_LEFT/' \
          'n015-2018-07-18-11-07-57+0800__CAM_BACK_LEFT__1531883530447423.jpg'
    ann_file = 'tests/data/nuscenes/nus_infos_mono3d.coco.json'
    detector_cfg = 'configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_' \
                   '2x8_1x_nus-mono3d.py'
    detector = init_model(detector_cfg, device='cuda:0')
    results = inference_mono_3d_detector(detector, img, ann_file)
    bboxes_3d = results[0][0]['img_bbox']['boxes_3d']
    scores_3d = results[0][0]['img_bbox']['scores_3d']
    labels_3d = results[0][0]['img_bbox']['labels_3d']
    assert bboxes_3d.tensor.shape[0] >= 0
    assert bboxes_3d.tensor.shape[1] == 9
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0


def test_inference_segmentor():
    # PN2 only has GPU implementations
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    pcd = 'tests/data/scannet/points/scene0000_00.bin'
    segmentor_cfg = 'configs/pointnet2/pointnet2_ssg_' \
                    '16x2_cosine_200e_scannet_seg-3d-20class.py'
    segmentor = init_model(segmentor_cfg, device='cuda:0')
    results = inference_segmentor(segmentor, pcd)
    seg_3d = results[0][0]['semantic_mask']
    assert seg_3d.shape == torch.Size([100])
    assert seg_3d.min() >= 0
    assert seg_3d.max() <= 19


def test_single_gpu_test():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    cfg = _get_config_module('votenet/votenet_16x8_sunrgbd-3d-10class.py')
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    dataset_cfg = cfg.data.test
    dataset_cfg.data_root = './tests/data/sunrgbd'
    dataset_cfg.ann_file = 'tests/data/sunrgbd/sunrgbd_infos.pkl'
    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    model = MMDataParallel(model, device_ids=[0])
    results = single_gpu_test(model, data_loader)
    bboxes_3d = results[0]['boxes_3d']
    scores_3d = results[0]['scores_3d']
    labels_3d = results[0]['labels_3d']
    assert bboxes_3d.tensor.shape[0] >= 0
    assert bboxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0
