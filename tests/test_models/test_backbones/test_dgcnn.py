# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmengine.device import is_musa_available, is_cuda_available

from mmdet3d.registry import MODELS


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not is_cuda_available(), reason="requires CUDA support"
            ),
        ),
        pytest.param(
            "musa",
            marks=pytest.mark.skipif(
                not is_musa_available(), reason="requires MUSA support"
            ),
        ),
    ],
)
def test_dgcnn_gf(device: str):
    if not is_cuda_available() and not is_musa_available():
        pytest.skip()

    # DGCNNGF used in segmentation
    cfg = dict(
        type='DGCNNBackbone',
        in_channels=6,
        num_samples=(20, 20, 20),
        knn_modes=['D-KNN', 'F-KNN', 'F-KNN'],
        radius=(None, None, None),
        gf_channels=((64, 64), (64, 64), (64, )),
        fa_channels=(1024, ),
        act_cfg=dict(type='ReLU'))

    self = MODELS.build(cfg)
    self.to(device)

    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 6).to(device)  # (B, N, 6)
    # test forward
    ret_dict = self(xyz)
    gf_points = ret_dict['gf_points']
    fa_points = ret_dict['fa_points']

    assert len(gf_points) == 4
    assert gf_points[0].shape == torch.Size([1, 100, 6])
    assert gf_points[1].shape == torch.Size([1, 100, 64])
    assert gf_points[2].shape == torch.Size([1, 100, 64])
    assert gf_points[3].shape == torch.Size([1, 100, 64])
    assert fa_points.shape == torch.Size([1, 100, 1216])
