import numpy as np
import torch
import trimesh
from mmcv.ops.nms import batched_nms
from mmcv.runner import BaseModule, force_fp32
from torch.nn import functional as F

from mmdet3d.core import xywhr2xyxyr
from mmdet3d.core.bbox.structures import (DepthInstance3DBoxes,
                                          LiDARInstance3DBoxes)
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models import HEADS, build_loss
from .base_separate_conv_bbox_head import BaseSeparateConvBboxHead


def write_ply(points, points_label, out_filename):
    """Write points into ``ply`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 3:
            c = [0, 0, 0]
            if points_label[i].astype(int) <= 2:
                c[points_label[i].astype(int)] = 1
            fout.write('v %f %f %f %d %d %d\n' %
                       (points[i, 0], points[i, 1], points[i, 2], c[0] * 255,
                        c[1] * 255, c[2] * 255))

        else:
            fout.write(
                'v %f %f %f %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], points_label[i]))
    fout.close()


def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes.

    Args:
        scene_bbox(list[ndarray] or ndarray): xyz pos of center and
            3 lengths (dx,dy,dz) and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename(str): Filename.
    """

    def heading2rotmat(heading_angle):
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    if len(scene_bbox) == 0:
        scene_bbox = np.zeros((1, 7))
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')

    return


@HEADS.register_module()
class PointRPNHead(BaseModule):

    def __init__(self,
                 num_classes,
                 num_dir_bins,
                 train_cfg,
                 test_cfg,
                 pred_layer_cfg=None,
                 cls_loss=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 bbox_loss=None,
                 corner_loss=None,
                 bbox_coder=None,
                 init_cfg=None,
                 pretrained=None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.num_dir_bins = num_dir_bins
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # build loss function
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.corner_loss = build_loss(corner_loss)

        # build box coder
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.conv_pred = BaseSeparateConvBboxHead(
            **pred_layer_cfg,
            num_cls_out_channels=self._get_cls_out_channels(),
            num_reg_out_channels=self._get_reg_out_channels())

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (1)
        return self.num_classes

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # Bbox classification and regression
        # (center residual (3), size regression (3)
        # torch.cos(yaw) (1), torch.sin(yaw)(1)
        return 3 + 3 + 2

    def forward(self, feat_dict):
        point_features = feat_dict['fp_features'][-1]
        point_cls_preds, point_box_preds = self.conv_pred(point_features)
        '''
        ret_dict = {
            'fp_points': feat_dict['fp_xyz'][-1],
            'fp_indices': feat_dict['fp_indices'][-1],
            'fp_features': feat_dict['fp_features'][-1]
        }
        print('preds: ', point_box_preds.shape, point_cls_preds.shape)
        roi_scores, roi_labels = torch.max(
            point_cls_preds.transpose(2,1),dim=2)
        print('result: ', roi_scores.shape, roi_labels.shape)
        bbox_preds = self.bbox_coder.decode(point_box_preds.transpose(2,1),
                                            feat_dict['fp_xyz'][-1],
                                            roi_labels)
        '''
        '''
        print(point_cls_preds.shape)
        points_label = point_cls_preds.transpose(2, 1). \
            reshape(-1).cpu().data.numpy()
        points_label[points_label>0.9] = 1
        points = feat_dict['fp_xyz'][-1].cpu().data.numpy()
        print(points_label.shape)
        print(points.shape)
        write_ply(points[0],points_label,'/tmp/label_points.obj')
        bbox3d = self.bbox_coder.decode(ret_dict)
        write_oriented_bbox(bbox3d[0].cpu().numpy(),'/tmp/label_bboxes.ply')
        # assert 0
        '''
        return (point_box_preds.transpose(2,
                                          1), point_cls_preds.transpose(2, 1))

    @force_fp32(apply_to=('bbox_preds'))
    def loss(self,
             bbox_preds,
             cls_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             img_metas=None):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of PointRCNNHead.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            img_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses of PointRCNN.
        """
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   bbox_preds)
        (bbox_targets, mask_targets, positive_mask, negative_mask,
         box_loss_weights, corner3d_targets, point_targets) = targets

        # bbox loss
        bbox_loss = self.bbox_loss(bbox_preds, bbox_targets,
                                   box_loss_weights.unsqueeze(-1))
        '''
        # corner loss
        pred_bbox3d = self.bbox_coder.decode(bbox_preds, point_targets,
                                             mask_targets)
        pred_bbox3d = pred_bbox3d.reshape(-1, pred_bbox3d.shape[-1])
        pred_bbox3d = img_metas[0]['box_type_3d'](
            pred_bbox3d.clone(),
            box_dim=pred_bbox3d.shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5))
        pred_corners3d = pred_bbox3d.corners.reshape(-1, 8, 3)
        corner_loss = self.corner_loss(
            pred_corners3d,
            corner3d_targets.reshape(-1, 8, 3),
            weight=box_loss_weights.view(-1, 1, 1))
        '''
        # calculate semantic loss
        semantic_points = cls_preds.reshape(-1, self.num_classes)
        semantic_targets = mask_targets
        semantic_targets[negative_mask] = self.num_classes
        semantic_points_label = semantic_targets
        # for ignore, but now we do not have ignore label
        # semantic_loss_weight = negative_mask.float() + positive_mask.float()
        semantic_loss = self.cls_loss(semantic_points,
                                      semantic_points_label.reshape(-1))
        semantic_loss /= positive_mask.float().sum()
        '''
        indices_xxx = 0
        points_label = semantic_points_label[indices_xxx].cpu().data.numpy()
        points = points[indices_xxx][:,0:3].cpu().data.numpy()
        write_ply(points,points_label,'/tmp/label_points_rpn.obj')
        gt_bboxes = gt_bboxes_3d[indices_xxx].tensor.cpu().numpy()
        gt_bboxes[:, 6] = -gt_bboxes[:, 6]
        write_oriented_bbox(gt_bboxes,'/tmp/label_bboxes_rpn.ply')
        assert 0
        points_mask = points_label==0
        pred_bbox3d = pred_bbox3d[0:16384][points_mask].tensor
        pred_bbox3d = pred_bbox3d.tensor.detach().cpu().numpy()
        pred_bbox3d[:, 6] = -pred_bbox3d[:, 6]
        write_oriented_bbox(pred_bbox3d, '/tmp/label_bboxes_pred_rpn.ply')

        assert 0
        '''

        losses = dict(bbox_loss=bbox_loss, semantic_loss=semantic_loss)
        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of ssd3d head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of ssd3d head.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        # find empty example
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)

        (bbox_targets, mask_targets, positive_mask, negative_mask,
         corner3d_targets,
         point_targets) = multi_apply(self.get_targets_single, points,
                                      gt_bboxes_3d, gt_labels_3d)

        bbox_targets = torch.stack(bbox_targets)
        corner3d_targets = torch.stack(corner3d_targets)
        mask_targets = torch.stack(mask_targets)
        positive_mask = torch.stack(positive_mask)
        negative_mask = torch.stack(negative_mask)
        point_targets = torch.stack(point_targets)
        box_loss_weights = positive_mask / (positive_mask.sum() + 1e-6)

        return (bbox_targets, mask_targets, positive_mask, negative_mask,
                box_loss_weights, corner3d_targets, point_targets)

    def get_targets_single(self, points, gt_bboxes_3d, gt_labels_3d):
        """Generate targets of ssd3d head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.
            seed_points (torch.Tensor): Seed points of candidate points.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]

        # transform the bbox coordinate to the pointcloud coordinate
        gt_corner3d = gt_bboxes_3d.corners
        gt_bboxes_3d_tensor = gt_bboxes_3d.tensor.clone()
        gt_bboxes_3d_tensor[..., 2] += gt_bboxes_3d_tensor[..., 5] / 2

        points_mask, assignment = self._assign_targets_by_points_inside(
            gt_bboxes_3d, points)
        gt_bboxes_3d_tensor = gt_bboxes_3d_tensor[assignment]
        mask_targets = gt_labels_3d[assignment]
        corner3d_targets = gt_corner3d[assignment]

        bbox_targets = self.bbox_coder.encode(gt_bboxes_3d_tensor,
                                              points[..., 0:3], mask_targets)

        positive_mask = (points_mask.max(1)[0] > 0)
        negative_mask = (points_mask.max(1)[0] == 0)
        point_targets = points[..., 0:3]
        return (bbox_targets, mask_targets, positive_mask, negative_mask,
                corner3d_targets, point_targets)

    def get_bboxes(self,
                   points,
                   bbox_preds,
                   cls_preds,
                   input_metas,
                   training_flag=False,
                   rescale=False):
        """Generate bboxes from sdd3d head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from PointRCNN head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # decode boxes
        # sem_scores = F.sigmoid(bbox_preds['obj_scores']).transpose(1, 2)
        # obj_scores = sem_scores.max(-1)[0]
        sem_scores = F.sigmoid(cls_preds)
        obj_scores = sem_scores.max(-1)[0]
        object_class = torch.argmax(sem_scores, -1)

        batch_size = sem_scores.shape[0]
        results = list()
        for b in range(batch_size):
            bbox3d = self.bbox_coder.decode(bbox_preds[b], points[b, ..., :3],
                                            object_class[b])
            bbox_selected, score_selected, labels, cls_preds_selected = \
                self.multiclass_nms_single(obj_scores[b], sem_scores[b],
                                           bbox3d, points[b, ..., :3],
                                           input_metas[b], training_flag)
            bbox = input_metas[b]['box_type_3d'](
                bbox_selected.clone(),
                box_dim=bbox_selected.shape[-1],
                with_yaw=True)
            results.append((bbox, score_selected, labels, cls_preds_selected))
        return results

    def class_agnostic_nms(self, obj_scores, sem_scores, bbox, points,
                           input_meta, training_flag):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5))
        bbox_for_nms = xywhr2xyxyr(bbox.bev)

        bbox_selected = []
        score_selected = []
        labels = []
        cls_preds = []

        num_rpn_proposal = self.test_cfg.max_output_num
        score_thr = self.test_cfg.score_thr
        if training_flag:
            num_rpn_proposal = self.train_cfg.rpn_proposal.max_num
            score_thr = self.train_cfg.rpn_proposal.score_thr

        score_thr_inds = obj_scores > score_thr
        _scores = obj_scores[score_thr_inds]
        _bboxes_for_nms = bbox_for_nms[score_thr_inds, :]
        _bboxes = bbox[score_thr_inds]
        _sem_scores = sem_scores[score_thr_inds]
        _classes = torch.argmax(_sem_scores, -1)

        selected = nms_gpu(_bboxes_for_nms, _scores, score_thr)
        if selected.shape[0] > num_rpn_proposal:
            selected = selected[:num_rpn_proposal]

        if len(selected) > 0:
            bbox_selected = _bboxes[selected].tensor
            score_selected = _scores[selected]
            labels = _classes[selected]
            cls_preds = _sem_scores[selected]

        return bbox_selected, score_selected, labels, cls_preds

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta, training_flag):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        num_bbox = bbox.shape[0]
        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5))

        if isinstance(bbox, LiDARInstance3DBoxes):
            box_idx = bbox.points_in_boxes(points)
            box_indices = box_idx.new_zeros([num_bbox + 1])
            box_idx[box_idx == -1] = num_bbox
            box_indices.scatter_add_(0, box_idx.long(),
                                     box_idx.new_ones(box_idx.shape))
            box_indices = box_indices[:-1]
            nonempty_box_mask = box_indices >= 0
        elif isinstance(bbox, DepthInstance3DBoxes):
            box_indices = bbox.points_in_boxes(points)
            nonempty_box_mask = box_indices.T.sum(1) >= 0
        else:
            raise NotImplementedError('Unsupported bbox type!')

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        num_rpn_proposal = self.test_cfg.max_output_num
        nms_cfg = self.test_cfg.nms_cfg
        score_thr = self.test_cfg.score_thr
        if training_flag:
            num_rpn_proposal = self.train_cfg.rpn_proposal.max_num
            nms_cfg = self.train_cfg.rpn_proposal.nms_cfg
            score_thr = self.train_cfg.rpn_proposal.score_thr

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = batched_nms(
            minmax_box3d[nonempty_box_mask][:, [0, 1, 3, 4]].detach(),
            obj_scores[nonempty_box_mask].detach(),
            bbox_classes[nonempty_box_mask].detach(), nms_cfg, True)[1]

        if nms_selected.shape[0] > num_rpn_proposal:
            nms_selected = nms_selected[:num_rpn_proposal]

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores >= score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels, cls_preds = [], [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected])
                cls_preds.append(sem_scores[selected])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
            cls_preds = torch.cat(cls_preds, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]
            cls_preds = sem_scores[selected]

        return bbox_selected, score_selected, labels, cls_preds

    def _assign_targets_by_points_inside(self, bboxes_3d, points):
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (BaseInstance3DBoxes): Instance of bounding boxes.
            points (torch.Tensor): Points of a batch.

        Returns:
            tuple[torch.Tensor]: Flags indicating whether each point is
                inside bbox and the index of box where each point are in.
        """
        # TODO: align points_in_boxes function in each box_structures
        num_bbox = bboxes_3d.tensor.shape[0]
        if isinstance(bboxes_3d, LiDARInstance3DBoxes):
            assignment = bboxes_3d.points_in_boxes(points[:, 0:3]).long()
            points_mask = assignment.new_zeros(
                [assignment.shape[0], num_bbox + 1])
            assignment[assignment == -1] = num_bbox
            points_mask.scatter_(1, assignment.unsqueeze(1), 1)
            points_mask = points_mask[:, :-1]
            assignment[assignment == num_bbox] = num_bbox - 1
        elif isinstance(bboxes_3d, DepthInstance3DBoxes):
            points_mask = bboxes_3d.points_in_boxes(points)
            assignment = points_mask.argmax(dim=-1)
        else:
            raise NotImplementedError('Unsupported bbox type!')

        return points_mask, assignment
