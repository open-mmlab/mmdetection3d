import mmcv
import pytest
import torch

from mmdet3d.core import merge_aug_bboxes_3d
from mmdet3d.core.bbox import DepthInstance3DBoxes


def test_merge_aug_bboxes_3d():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    img_meta = dict(
        pcd_horizontal_flip=False,
        pcd_vertical_flip=False,
        pcd_scale_factor=1.0)
    img_metas = [[img_meta]]
    boxes_3d = DepthInstance3DBoxes(
        torch.tensor(
            [[1.0473, 4.1687, -1.2317, 2.3021, 1.8876, 1.9696, 1.6956],
             [2.5831, 4.8117, -1.2733, 0.5852, 0.8832, 0.9733, 1.6500],
             [-1.0864, 1.9045, -1.2000, 0.7128, 1.5631, 2.1045, 0.1022]],
            device='cuda'))
    labels_3d = torch.tensor([0, 7, 6])
    scores_3d = torch.tensor([0.5, 1.0, 1.0])
    aug_result = dict(
        boxes_3d=boxes_3d, labels_3d=labels_3d, scores_3d=scores_3d)
    aug_results = [aug_result]
    test_cfg = mmcv.ConfigDict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50)
    results = merge_aug_bboxes_3d(aug_results, img_metas, test_cfg)
    expected_boxes_3d = torch.tensor(
        [[-1.0864, 1.9045, -1.2000, 0.7128, 1.5631, 2.1045, 0.1022],
         [2.5831, 4.8117, -1.2733, 0.5852, 0.8832, 0.9733, 1.6500],
         [1.0473, 4.1687, -1.2317, 2.3021, 1.8876, 1.9696, 1.6956]])
    expected_scores_3d = torch.tensor([1.0000, 1.0000, 0.5000])
    expected_labels_3d = torch.tensor([6, 7, 0])
    assert torch.allclose(results['boxes_3d'].tensor, expected_boxes_3d)
    assert torch.allclose(results['scores_3d'], expected_scores_3d)
    assert torch.all(results['labels_3d'] == expected_labels_3d)
