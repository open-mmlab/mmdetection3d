import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet.models.dense_heads import DETRHead
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.utils import multi_apply
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS, TASK_UTILS
from .util import normalize_bbox


@MODELS.register_module()
class DETR3DHead(DETRHead):
    """Head of DETR3D.

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bbox_coder (obj:`ConfigDict`): Configs to build the bbox coder
        num_cls_fcs (int) : the number of layers in cls and reg branch
        code_weights (List[double]) : loss weights of
            (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y)
        code_size (int) : size of code_weights
    """

    def __init__(
            self,
            *args,
            with_box_refine=False,
            as_two_stage=False,
            transformer=None,
            bbox_coder=None,
            num_cls_fcs=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            code_size=10,
            **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.code_size = code_size
        self.code_weights = code_weights

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1
        super(DETR3DHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        # DETR sampling=False, so use PseudoSampler, format the result
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = TASK_UTILS.build(sampler_cfg)

        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False),
            requires_grad=False)

    # forward_train -> loss
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats: List[Tensor], img_metas: List[Dict],
                **kwargs) -> Dict[str, Tensor]:
        """Forward function.

        Args:
            mlvl_feats (List[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format
                (cx, cy, l, w, cz, h, sin(φ), cos(φ), vx, vy).
                Shape [nb_dec, bs, num_query, 10].
        """
        query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            img_metas=img_metas,
            **kwargs)
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])  # shape: ([B, num_q, 10])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            tmp[..., 0:1] = \
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) \
                + self.pc_range[0]
            tmp[..., 1:2] = \
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) \
                + self.pc_range[1]
            tmp[..., 4:5] = \
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) \
                + self.pc_range[2]

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def _get_target_single(
            self,
            cls_score: Tensor,  # [query, num_cls]
            bbox_pred: Tensor,  # [query, 10]
            gt_instances_3d: InstanceList) -> Tuple[Tensor, ...]:
        """Compute regression and classification targets for a single image."""
        # turn bottm center into gravity center
        gt_bboxes = gt_instances_3d.bboxes_3d  # [num_gt, 9]
        gt_bboxes = torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)

        gt_labels = gt_instances_3d.labels_3d  # [num_gt, num_cls]
        # assigner and sampler: PseudoSampler
        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore=None)
        sampling_result = self.sampler.sample(
            assign_result, InstanceData(priors=bbox_pred),
            InstanceData(bboxes_3d=gt_bboxes))
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        num_bboxes = bbox_pred.size(0)
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # theta in gt_bbox here is still a single scalar
        bbox_targets = torch.zeros_like(bbox_pred)[..., :self.code_size - 1]
        bbox_weights = torch.zeros_like(bbox_pred)
        # only matched query will learn from bbox coord
        bbox_weights[pos_inds] = 1.0

        # fix empty gt bug in multi gpu training
        if sampling_result.pos_gt_bboxes.shape[0] == 0:
            sampling_result.pos_gt_bboxes = \
                sampling_result.pos_gt_bboxes.reshape(0, self.code_size - 1)

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def get_targets(
            self,
            batch_cls_scores: List[Tensor],  # bs[num_q,num_cls]
            batch_bbox_preds: List[Tensor],  # bs[num_q,10]
            batch_gt_instances_3d: InstanceList) -> tuple():
        """"Compute regression and classification targets for a batch image for
        a single decoder layer.

        Args:
            batch_cls_scores (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            batch_bbox_preds (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y) and
                shape [num_query, 10]
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes_3d``、``labels_3d``.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_target_single,
                                                     batch_cls_scores,
                                                     batch_bbox_preds,
                                                     batch_gt_instances_3d)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_by_feat_single(
        self,
        batch_cls_scores: Tensor,  # bs,num_q,num_cls
        batch_bbox_preds: Tensor,  # bs,num_q,10
        batch_gt_instances_3d: InstanceList
    ) -> Tuple[Tensor, Tensor]:
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
           batch_cls_scores (Tensor): Box score logits from a single
                decoder layer for batched images with shape [num_query,
                cls_out_channels].
            batch_bbox_preds (Tensor): Sigmoid outputs from a single
                decoder layer for batched images, with normalized coordinate
                (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y) and
                shape [num_query, 10]
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d. It usually has ``bboxes_3d``,``labels_3d``.
        Returns:
            tulple(Tensor, Tensor): cls and reg loss for outputs from
                a single decoder layer.
        """
        batch_size = batch_cls_scores.size(0)  # batch size
        cls_scores_list = [batch_cls_scores[i] for i in range(batch_size)]
        bbox_preds_list = [batch_bbox_preds[i] for i in range(batch_size)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances_3d)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        batch_cls_scores = batch_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                batch_cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            batch_cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        batch_bbox_preds = batch_bbox_preds.reshape(-1,
                                                    batch_bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        # neg_query is all 0, log(0) is NaN
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            batch_bbox_preds[isnotnan, :self.code_size],
            normalized_bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    # original loss()
    def loss_by_feat(
            self,
            batch_gt_instances_3d: InstanceList,
            preds_dicts: Dict[str, Tensor],
            batch_gt_instances_3d_ignore: OptInstanceList = None) -> Dict:
        """Compute loss of the head.

        Args:
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
                gt_instance.  It usually includes ``bboxes``、``labels``.
            batch_gt_instances_3d_ignore (list[:obj:`InstanceData`], Optional):
                NOT supported.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_3d_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for batch_gt_instances_3d_ignore setting to None.'
        all_cls_scores = preds_dicts[
            'all_cls_scores']  # num_dec,bs,num_q,num_cls
        all_bbox_preds = preds_dicts['all_bbox_preds']  # num_dec,bs,num_q,10
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        # calculate loss for each decoder layer
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_3d_list = [
            batch_gt_instances_3d for _ in range(num_dec_layers)
        ]
        losses_cls, losses_bbox = multi_apply(self.loss_by_feat_single,
                                              all_cls_scores, all_bbox_preds,
                                              batch_gt_instances_3d_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            enc_loss_cls, enc_losses_bbox = self.loss_by_feat_single(
                enc_cls_scores, enc_bbox_preds, batch_gt_instances_3d_list)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    def predict_by_feat(self,
                        preds_dicts,
                        img_metas,
                        rescale=False) -> InstanceList:
        """Transform network output for a batch into bbox predictions.

        Args:
            preds_dicts (Dict[str, Tensor]):
                -all_cls_scores (Tensor): Outputs from the classification head,
                    shape [nb_dec, bs, num_query, cls_out_channels]. Note
                    cls_out_channels should includes background.
                -all_bbox_preds (Tensor): Sigmoid outputs from the regression
                    head with normalized coordinate format
                    (cx, cy, l, w, cz, h, rot_sine, rot_cosine, v_x, v_y).
                    Shape [nb_dec, bs, num_query, 10].
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                  (num_instances, C), where C >= 7.
        """
        # sinθ & cosθ ---> θ
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)  # batch size
        ret_list = []
        for i in range(num_samples):
            results = InstanceData()
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, self.code_size - 1)

            results.bboxes_3d = bboxes
            results.scores_3d = preds['scores']
            results.labels_3d = preds['labels']
            ret_list.append(results)
        return ret_list
