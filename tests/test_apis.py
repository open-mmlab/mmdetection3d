import pytest
import torch

from mmdet3d.apis import inference_detector, init_detector


def test_inference_detector():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    pcd = 'tests/data/sunrgbd/points/000001.bin'
    detector_cfg = 'configs/votenet/votenet_16x8_sunrgbd-3d-10class.py'
    detector = init_detector(detector_cfg)
    results = inference_detector(detector, pcd)
    bboxes_3d = results[0]['boxes_3d']
    scores_3d = results[0]['scores_3d']
    labels_3d = results[0]['labels_3d']
    assert bboxes_3d.tensor.shape[0] >= 0
    assert bboxes_3d.tensor.shape[1] == 7
    assert scores_3d.shape[0] >= 0
    assert labels_3d.shape[0] >= 0
