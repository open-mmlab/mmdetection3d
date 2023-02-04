# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from .two_stage import TwoStage3DDetector


@MODELS.register_module()
class H3DNet(TwoStage3DDetector):
    r"""H3DNet model.

    Please refer to the `paper <https://arxiv.org/abs/2006.05682>`_

    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        rpn_head (dict, optional): Config dict of rpn head. Defaults to None.
        roi_head (dict, optional): Config dict of roi head. Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        init_cfg (dict, optional): the config to control the
           initialization. Default to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 rpn_head: Optional[dict] = None,
                 roi_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 **kwargs) -> None:
        super(H3DNet, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs)

    def extract_feat(self, batch_inputs_dict: dict) -> None:
        """Directly extract features from the backbone+neck.

        Args:

            batch_inputs_dict (dict): The model input dict which include
                'points'.

                - points (list[torch.Tensor]): Point cloud of each sample.

        Returns:
            dict: Dict of feature.
        """
        stack_points = torch.stack(batch_inputs_dict['points'])
        x = self.backbone(stack_points)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs_dict: Dict[str, Union[List, Tensor]],
             batch_data_samples: List[Det3DDataSample], **kwargs) -> dict:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        feats_dict = self.extract_feat(batch_inputs_dict)

        feats_dict['fp_xyz'] = [feats_dict['fp_xyz_net0'][-1]]
        feats_dict['fp_features'] = [feats_dict['hd_feature']]
        feats_dict['fp_indices'] = [feats_dict['fp_indices_net0'][-1]]

        losses = dict()
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            # note, the feats_dict would be added new key & value in rpn_head
            rpn_losses, rpn_proposals = self.rpn_head.loss_and_predict(
                batch_inputs_dict['points'],
                feats_dict,
                batch_data_samples,
                ret_target=True,
                proposal_cfg=proposal_cfg)
            feats_dict['targets'] = rpn_losses.pop('targets')
            losses.update(rpn_losses)
            feats_dict['rpn_proposals'] = rpn_proposals
        else:
            raise NotImplementedError

        roi_losses = self.roi_head.loss(batch_inputs_dict['points'],
                                        feats_dict, batch_data_samples,
                                        **kwargs)
        losses.update(roi_losses)

        return losses

    def predict(
            self, batch_input_dict: Dict,
            batch_data_samples: List[Det3DDataSample]
    ) -> List[Det3DDataSample]:
        """Get model predictions.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
                contains the meta information of each sample and
                corresponding annotations.

        Returns:
            list: Predicted 3d boxes.
        """

        feats_dict = self.extract_feat(batch_input_dict)
        feats_dict['fp_xyz'] = [feats_dict['fp_xyz_net0'][-1]]
        feats_dict['fp_features'] = [feats_dict['hd_feature']]
        feats_dict['fp_indices'] = [feats_dict['fp_indices_net0'][-1]]

        if self.with_rpn:
            proposal_cfg = self.test_cfg.rpn
            rpn_proposals = self.rpn_head.predict(
                batch_input_dict['points'],
                feats_dict,
                batch_data_samples,
                use_nms=proposal_cfg.use_nms)
            feats_dict['rpn_proposals'] = rpn_proposals
        else:
            raise NotImplementedError

        results_list = self.roi_head.predict(
            batch_input_dict['points'],
            feats_dict,
            batch_data_samples,
            suffix='_optimized')
        return self.add_pred_to_datasample(batch_data_samples, results_list)
