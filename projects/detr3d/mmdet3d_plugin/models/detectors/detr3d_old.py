from typing import Dict, List, Optional, Sequence

import numpy as np
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from torch import Tensor

from .detr3d import Detr3D


@MODELS.register_module()
class Detr3D_old(Detr3D):
    """Detr3D for old models trained earlier than mmdet3d-1.0.0."""

    def __init__(self, **kawrgs):
        super().__init__(**kawrgs)

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_metas = self.add_lidar2img(batch_input_metas)
        img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        outs = self.pts_bbox_head(img_feats, batch_input_metas)

        results_list_3d = self.pts_bbox_head.predict_by_feat(
            outs, batch_input_metas, **kwargs)

        # change the bboxes' format
        for item in results_list_3d:
            #cx, cy, cz, w, l, h, rot, vx, vy
            item.bboxes_3d.tensor[..., [3, 4]] = item.bboxes_3d.tensor[...,
                                                                       [4, 3]]
            item.bboxes_3d.tensor[
                ..., 6] = -item.bboxes_3d.tensor[..., 6] - np.pi / 2

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d)
        return detsamples
