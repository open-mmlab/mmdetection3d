import pytest
import torch
from mmcv.parallel import MMDataParallel
from os.path import dirname, exists, join

from mmdet3d.apis import inference_detector, init_detector, single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
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
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
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
