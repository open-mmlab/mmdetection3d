import torch

from mmdet3d.core import draw_heatmap_gaussian


def test_gaussian():
    heatmap = torch.zeros((128, 128))
    ct_int = torch.tensor([64, 64], dtype=torch.int32)
    radius = 2
    draw_heatmap_gaussian(heatmap, ct_int, radius)
    assert torch.isclose(torch.sum(heatmap), torch.tensor(4.3505), atol=1e-3)
