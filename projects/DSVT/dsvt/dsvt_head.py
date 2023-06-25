from typing import Dict, List, Tuple

import torch
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models import CenterHead
from mmdet3d.models.layers import circle_nms, nms_bev
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, xywhr2xyxyr


@MODELS.register_module()
class DSVTCenterHead(CenterHead):
    """CenterHead for DSVT.

    This head adds IoU prediction branch based on the original CenterHead.
    """

    def __init__(self, *args, **kwargs):
        super(DSVTCenterHead, self).__init__(*args, **kwargs)

    def forward_single(self, x: Tensor) -> dict:
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def forward(self, feats: List[Tensor]) -> Tuple[List[Tensor]]:
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return multi_apply(self.forward_single, feats)

    def loss(self, pts_feats: List[Tensor],
             batch_data_samples: List[Det3DDataSample], *args,
             **kwargs) -> Dict[str, Tensor]:
        """Forward function of training.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict: Losses of each branch.
        """
        pass

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        """Loss function for CenterHead.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results of
                multiple tasks. The outer tuple indicate  different
                tasks head, and the internal list indicate different
                FPN level.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and\
                ``labels_3d`` attributes.

        Returns:
            dict[str,torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        pass

    def predict(self,
                pts_feats: Tuple[torch.Tensor],
                batch_data_samples: List[Det3DDataSample],
                rescale=True,
                **kwargs) -> List[InstanceData]:
        """
        Args:
            pts_feats (Tuple[torch.Tensor]): Point features..
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes meta information of data.
            rescale (bool): Whether rescale the resutls to
                the original scale.

        Returns:
            list[:obj:`InstanceData`]: List of processed predictions. Each
            InstanceData contains 3d Bounding boxes and corresponding
            scores and labels.
        """
        preds_dict = self(pts_feats)
        batch_size = len(batch_data_samples)
        batch_input_metas = []
        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            batch_input_metas.append(metainfo)

        results_list = self.predict_by_feat(
            preds_dict, batch_input_metas, rescale=rescale, **kwargs)
        return results_list

    def predict_by_feat(self, preds_dicts: Tuple[List[dict]],
                        batch_input_metas: List[dict], *args,
                        **kwargs) -> List[InstanceData]:
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results of
                multiple tasks. The outer tuple indicate  different
                tasks head, and the internal list indicate different
                FPN level.
            batch_input_metas (list[dict]): Meta info of multiple
                inputs.

        Returns:
            list[:obj:`InstanceData`]: Instance prediction
            results of each sample after the post process.
            Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (:obj:`LiDARInstance3DBoxes`): Prediction
                  of bboxes, contains a tensor with shape
                  (num_instances, 7) or (num_instances, 9), and
                  the last 2 dimensions of 9 is
                  velocity.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rotc = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rots = preds_dict[0]['rot'][:, 1].unsqueeze(1)
            batch_iou = (preds_dict[0]['iou'] +
                         1) * 0.5 if 'iou' in preds_dict[0] else None

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                iou=batch_iou)
            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds, batch_cls_preds, batch_cls_labels, batch_iou_preds = [], [], [], []  # noqa: E501
            for box in temp:
                batch_reg_preds.append(box['bboxes'])
                batch_cls_preds.append(box['scores'])
                batch_cls_labels.append(box['labels'].long())
                batch_iou_preds.append(box['iou'])
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(task_id, num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_iou_preds, batch_cls_labels,
                                             batch_input_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            temp_instances = InstanceData()
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = batch_input_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            temp_instances.bboxes_3d = bboxes
            temp_instances.scores_3d = scores
            temp_instances.labels_3d = labels
            ret_list.append(temp_instances)
        return ret_list

    def get_task_detections(self, task_id, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_iou_preds, batch_cls_labels,
                            img_metas):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_iou_preds (list[torch.Tensor]): Prediction IoU with the
                shape of [N].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        for i, (box_preds, cls_preds, iou_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_iou_preds,
                    batch_cls_labels)):
            pred_iou = torch.clamp(iou_preds, min=0, max=1.0)
            iou_rectifier = pred_iou.new_tensor(
                self.test_cfg['iou_rectifier'][task_id])
            cls_preds = torch.pow(cls_preds,
                                  1 - iou_rectifier[cls_labels]) * torch.pow(
                                      pred_iou, iou_rectifier[cls_labels])

            # Apply NMS in bird eye view
            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds

            if top_scores.shape[0] != 0:
                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)

                pre_max_size = self.test_cfg['pre_max_size'][task_id]
                post_max_size = self.test_cfg['post_max_size'][task_id]
                # cls_label_per_task = self.cls_id_mapping_per_task[task_id]
                all_selected_mask = torch.zeros_like(top_labels, dtype=bool)
                all_indices = torch.arange(top_labels.size(0)).to(
                    top_labels.device)
                # Mind this when training on the new coordinate
                # Transform to old mmdet3d coordinate
                boxes_for_nms[:, 4] = (-boxes_for_nms[:, 4] + torch.pi / 2 * 1)
                boxes_for_nms[:, 4] = (boxes_for_nms[:, 4] +
                                       torch.pi) % (2 * torch.pi) - torch.pi

                for i, nms_thr in enumerate(self.test_cfg['nms_thr'][task_id]):
                    label_mask = top_labels == i
                    selected = nms_bev(
                        boxes_for_nms[label_mask],
                        top_scores[label_mask],
                        thresh=nms_thr,
                        pre_max_size=pre_max_size[i],
                        post_max_size=post_max_size[i])
                    indices = all_indices[label_mask][selected]
                    all_selected_mask.scatter_(0, indices, True)
            else:
                all_selected_mask = []

            # if selected is not None:
            selected_boxes = box_preds[all_selected_mask]
            selected_labels = top_labels[all_selected_mask]
            selected_scores = top_scores[all_selected_mask]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                predictions_dict = dict(
                    bboxes=final_box_preds,
                    scores=final_scores,
                    labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
