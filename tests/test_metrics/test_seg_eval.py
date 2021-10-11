# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.core.evaluation.seg_eval import seg_eval


def test_indoor_eval():
    if not torch.cuda.is_available():
        pytest.skip()
    seg_preds = [
        torch.Tensor([
            0, 0, 1, 0, 0, 2, 1, 3, 1, 2, 1, 0, 2, 2, 2, 2, 1, 3, 0, 3, 3, 3, 3
        ])
    ]
    gt_labels = [
        torch.Tensor([
            0, 0, 0, 255, 0, 0, 1, 1, 1, 255, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3,
            3, 255
        ])
    ]

    label2cat = {
        0: 'car',
        1: 'bicycle',
        2: 'motorcycle',
        3: 'truck',
    }
    ret_value = seg_eval(gt_labels, seg_preds, label2cat, ignore_index=255)

    assert np.isclose(ret_value['car'], 0.428571429)
    assert np.isclose(ret_value['bicycle'], 0.428571429)
    assert np.isclose(ret_value['motorcycle'], 0.6666667)
    assert np.isclose(ret_value['truck'], 0.6666667)

    assert np.isclose(ret_value['acc'], 0.7)
    assert np.isclose(ret_value['acc_cls'], 0.7)
    assert np.isclose(ret_value['miou'], 0.547619048)
