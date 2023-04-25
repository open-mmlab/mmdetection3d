# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmengine.model import BaseTTAModel

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class Seg3DTTAModel(BaseTTAModel):

    def merge_preds(self, data_samples_list: List[SampleList]) -> SampleList:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (List[List[:obj:`Det3DDataSample`]]): List of
                predictions of all enhanced data.

        Returns:
            List[:obj:`Det3DDataSample`]: Merged prediction.
        """
        predictions = []
        for data_samples in data_samples_list:
            seg_logits = data_samples[0].pts_seg_logits.pts_seg_logits
            logits = torch.zeros(seg_logits.shape).to(seg_logits)
            for data_sample in data_samples:
                seg_logit = data_sample.pts_seg_logits.pts_seg_logits
                logits += seg_logit.softmax(dim=0)
            logits /= len(data_samples)
            seg_pred = logits.argmax(dim=0)
            data_samples[0].pred_pts_seg.pts_semantic_mask = seg_pred
            predictions.append(data_samples[0])
        return predictions
