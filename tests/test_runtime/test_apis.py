import numpy as np
import os
import pytest
import tempfile
import torch
from mmcv.parallel import MMDataParallel
from os.path import dirname, exists, join

from mmdet3d.apis import (convert_SyncBN, inference_detector, init_detector,
                          show_result_meshlab, single_gpu_test)
from mmdet3d.core import Box3DMode
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


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
    temp_out_dir = tempfile.mkdtemp()
    out_dir, file_name = show_result_meshlab(data, result, temp_out_dir)
    expected_outfile_ply = file_name + '_pred.ply'
    expected_outfile_obj = file_name + '_points.obj'
    expected_outfile_ply_path = os.path.join(out_dir, file_name,
                                             expected_outfile_ply)
    expected_outfile_obj_path = os.path.join(out_dir, file_name,
                                             expected_outfile_obj)
    assert os.path.exists(expected_outfile_ply_path)
    assert os.path.exists(expected_outfile_obj_path)
    os.remove(expected_outfile_obj_path)
    os.remove(expected_outfile_ply_path)
    os.removedirs(os.path.join(temp_out_dir, file_name))


def test_inference_detector():
    pcd = 'tests/data/kitti/training/velodyne_reduced/000000.bin'
    detector_cfg = 'configs/pointpillars/hv_pointpillars_secfpn_' \
                   '6x8_160e_kitti-3d-3class.py'
    detector = init_detector(detector_cfg, device='cpu')
    results = inference_detector(detector, pcd)
    bboxes_3d = results[0][0]['boxes_3d']
    scores_3d = results[0][0]['scores_3d']
    labels_3d = results[0][0]['labels_3d']
    assert bboxes_3d.tensor.shape[0] >= 0
    assert bboxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0


def test_single_gpu_test():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    cfg = _get_config_module('votenet/votenet_16x8_sunrgbd-3d-10class.py')
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
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
