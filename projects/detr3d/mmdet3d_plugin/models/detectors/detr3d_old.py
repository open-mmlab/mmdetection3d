from mmdet.models import DETECTORS
from .detr3d import Detr3D
import numpy as np

@DETECTORS.register_module()
class Detr3D_old(Detr3D):
    """Detr3D for old models trained earlier than mmdet3d-1.0.0"""
    def __init__(self, **kawrgs):
        super().__init__(**kawrgs)
    
    def simple_test_pts(self, x, img_metas, rescale=False):

        bbox_results = super().simple_test_pts(x, img_metas, rescale=rescale)
        for item in bbox_results:
            #cx, cy, cz, w, l, h, rot, vx, vy
            item['boxes_3d'].tensor[..., [3,4]] = item['boxes_3d'].tensor[..., [4,3]]
            item['boxes_3d'].tensor[..., 6] = -item['boxes_3d'].tensor[..., 6] - np.pi/2

        return bbox_results