import numpy as np
import torch
import trimesh
from torch.nn import functional as F

from mmdet3d.core import AssignResult
from mmdet3d.core.bbox import bbox3d2result, bbox3d2roi
from mmdet.core import build_assigner, build_sampler
from mmdet.models import HEADS
from ..builder import build_head, build_roi_extractor
from .base_3droi_head import Base3DRoIHead


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
            c = (points_label[i] * 255).astype(int)
            fout.write('v %f %f %f %d %d %d\n' %
                       (points[i, 0], points[i, 1], points[i, 2], c, 0, 0))

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

    # scene.add_geometry(convert_oriented_box_to_trimesh_fmt(scene_bbox))
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')

    return


@HEADS.register_module()
class PointRCNNROIHead(Base3DRoIHead):
    """PointRCNN roi head for PointRCNN.

    Args:
        semantic_head (ConfigDict): Config of semantic head.
        num_classes (int): The number of classes.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(self,
                 bbox_head,
                 num_classes=1,
                 depth_normalizer=70,
                 point_roi_extractor=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(PointRCNNROIHead, self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.num_classes = num_classes
        self.depth_normalizer = depth_normalizer

        if point_roi_extractor is not None:
            self.point_roi_extractor = build_roi_extractor(point_roi_extractor)

        self.init_assigner_sampler()

    def init_mask_head(self):
        """Initialize mask head, skip since ``PointRCNNROIHead`` does not have
        one."""
        pass

    def init_bbox_head(self, bbox_head):
        """Initialize box head."""
        self.bbox_head = build_head(bbox_head)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            if isinstance(self.train_cfg.assigner, dict):
                self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            elif isinstance(self.train_cfg.assigner, list):
                self.bbox_assigner = [
                    build_assigner(res) for res in self.train_cfg.assigner
                ]
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)

    def forward_train(self, feats_dict, input_metas, proposal_list,
                      gt_bboxes_3d, gt_labels_3d):
        """Training forward function of PartAggregationROIHead.

        Args:
            feats_dict (dict): Contains features from the first stage.
            imput_metas (list[dict]): Meta info of each input.
            proposal_list (list[dict]): Proposal information from rpn.
                The dictionary should contain the following keys:

                - boxes_3d (:obj:`BaseInstance3DBoxes`): Proposal bboxes
                - labels_3d (torch.Tensor): Labels of proposals
                - cls_preds (torch.Tensor): Original scores of proposals
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]):
                GT bboxes of each sample. The bboxes are encapsulated
                by 3D box structures.
            gt_labels_3d (list[LongTensor]): GT labels of each sample.

        Returns:
            dict: losses from each head.

                - loss_semantic (torch.Tensor): loss of semantic head
                - loss_bbox (torch.Tensor): loss of bboxes
        """
        features = feats_dict['features']
        points = feats_dict['points']
        point_scores = feats_dict['points_scores']
        rcnn_gt_bboxes_3d = list()
        rcnn_gt_labels_3d = list()
        for i in range(len(gt_bboxes_3d)):
            valid_gt = gt_labels_3d[i] != -1
            gt_bboxes = input_metas[i]['box_type_3d'](
                gt_bboxes_3d[i].tensor[valid_gt].clone(),
                box_dim=gt_bboxes_3d[i].tensor.shape[-1],
                with_yaw=True,
                origin=(0.5, 0.5, 0.5))
            rcnn_gt_bboxes_3d.append(gt_bboxes)
            rcnn_gt_labels_3d.append(gt_labels_3d[i][valid_gt].clone())
        '''
        points_t = points[0]
        points_t = points_t[:,0:3].cpu().data.numpy()
        points_label = point_scores[0].squeeze().cpu().data.numpy()
        bbox = gt_bboxes_3d[0].tensor.cpu().data.numpy()
        bbox_p = proposal_list[0]['boxes_3d'].tensor.cpu().data.numpy()
        bbox_p_s = proposal_list[0]['scores_3d'].cpu().data.numpy()
        print(bbox.shape, bbox_p.shape,points_label.shape)
        bbox_p = bbox_p[bbox_p_s>0.65]
        bbox_p[..., 6] = -bbox_p[..., 6]
        bbox[..., 6] = -bbox[..., 6]
        write_oriented_bbox(bbox,'/tmp/label_bboxes.ply')
        write_oriented_bbox(bbox_p,'/tmp/label_bboxes_pred.ply')
        write_ply(points_t,points_label,'/tmp/label_points.obj')
        assert 0
        '''

        losses = dict()
        sample_results = self._assign_and_sample(proposal_list,
                                                 rcnn_gt_bboxes_3d,
                                                 rcnn_gt_labels_3d)

        # concat the depth, semantic features and backbone features
        features = features.transpose(1, 2).contiguous()
        point_depths = points.norm(dim=2) / self.depth_normalizer - 0.5
        features_list = [
            point_scores.unsqueeze(2),
            point_depths.unsqueeze(2), features
        ]
        features = torch.cat(features_list, dim=2)

        bbox_results = self._bbox_forward_train(features, points,
                                                sample_results)
        losses.update(bbox_results['loss_bbox'])

        return losses

    def simple_test(self, feats_dict, img_metas, proposal_list, **kwargs):
        """Simple testing forward function of PartAggregationROIHead.

        Note:
            This function assumes that the batch size is 1

        Args:
            feats_dict (dict): Contains features from the first stage.
            obj_scores (dict): Contains object scores from the first stage.
            img_metas (list[dict]): Meta info of each image.
            proposal_list (list[dict]): Proposal information from rpn.

        Returns:
            dict: Bbox results of one frame.
        """
        rois = bbox3d2roi([res['boxes_3d'].tensor for res in proposal_list])
        labels_3d = [res['labels_3d'] for res in proposal_list]

        features = feats_dict['features']
        points = feats_dict['points']
        point_scores = feats_dict['points_scores']

        features = features.transpose(1, 2).contiguous()
        point_depths = points.norm(dim=2) / self.depth_normalizer - 0.5
        features_list = [
            point_scores.unsqueeze(2),
            point_depths.unsqueeze(2), features
        ]

        features = torch.cat(features_list, dim=2)
        batch_size = features.shape[0]
        bbox_results = self._bbox_forward(features, points, batch_size, rois)

        bbox_list = self.bbox_head.get_bboxes(
            rois,
            bbox_results['bbox_pred'],
            labels_3d,
            torch.sigmoid(bbox_results['cls_score']),
            img_metas,
            cfg=self.test_cfg)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def _bbox_forward_train(self, global_feats, local_feats, sampling_results):
        """Forward training function of roi_extractor and bbox_head.

        Args:
            global_feats (torch.Tensor): global Point-wise semantic features.
            local_feats (torch.Tensor): local Point-wise semantic features.
            sampling_results (:obj:`SamplingResult`): Sampled results used
                for training.

        Returns:
            dict: Forward results including losses and predictions.
        """
        rois = bbox3d2roi([res.bboxes for res in sampling_results])
        batch_size = global_feats.shape[0]
        bbox_results = self._bbox_forward(global_feats, local_feats,
                                          batch_size, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, global_feats, points, batch_size, rois):
        """Forward function of roi_extractor and bbox_head used in both
        training and testing.

        Args:
            global_feats (torch.Tensor): Point-wise semantic features.
            local_feats (torch.Tensor): Point-wise part prediction features.
            rois (Tensor): Roi boxes.

        Returns:
            dict: Contains predictions of bbox_head and
                features of roi_extractor.
        """
        '''
        print('points shape: ', points.squeeze().shape)
        print('rois shape: ', rois.shape)
        points_t = points[0].clone()
        points_t = points_t[:,0:3].cpu().data.numpy()
        points_label = global_feats[0][..., 0].clone()
        points_label = points_label.cpu().data.numpy()
        print(points_label)
        bbox = rois[..., 1:].cpu().data.numpy()
        bbox[..., 2] += bbox[..., 5] / 2
        if torch.max( global_feats[0][..., 0]) > 0.9:
            write_oriented_bbox(bbox,'/tmp/label_bboxes.ply')
            write_ply(points_t,points_label,'/tmp/label_points.obj')
        '''
        pooled_point_feats = self.point_roi_extractor(global_feats, points,
                                                      batch_size, rois)

        cls_score, bbox_pred = self.bbox_head(pooled_point_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def _assign_and_sample(self, proposal_list, gt_bboxes_3d, gt_labels_3d):
        """Assign and sample proposals for training.

        Args:
            proposal_list (list[dict]): Proposals produced by RPN.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels

        Returns:
            list[:obj:`SamplingResult`]: Sampled results of each training
                sample.
        """
        sampling_results = []
        # bbox assign
        for batch_idx in range(len(proposal_list)):
            cur_proposal_list = proposal_list[batch_idx]
            cur_boxes = cur_proposal_list['boxes_3d']
            cur_labels_3d = cur_proposal_list['labels_3d']
            cur_gt_bboxes = gt_bboxes_3d[batch_idx].to(cur_boxes.device)
            cur_gt_labels = gt_labels_3d[batch_idx]
            batch_num_gts = 0
            # 0 is bg
            batch_gt_indis = cur_gt_labels.new_full((len(cur_boxes), ), 0)
            batch_max_overlaps = cur_boxes.tensor.new_zeros(len(cur_boxes))
            # -1 is bg
            batch_gt_labels = cur_gt_labels.new_full((len(cur_boxes), ), -1)

            # each class may have its own assigner
            if isinstance(self.bbox_assigner, list):
                for i, assigner in enumerate(self.bbox_assigner):
                    gt_per_cls = (cur_gt_labels == i)
                    pred_per_cls = (cur_labels_3d == i)
                    cur_assign_res = assigner.assign(
                        cur_boxes.tensor[pred_per_cls],
                        cur_gt_bboxes.tensor[gt_per_cls],
                        gt_labels=cur_gt_labels[gt_per_cls])
                    # gather assign_results in different class into one result
                    batch_num_gts += cur_assign_res.num_gts
                    # gt inds (1-based)
                    gt_inds_arange_pad = gt_per_cls.nonzero(
                        as_tuple=False).view(-1) + 1
                    # pad 0 for indice unassigned
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=0)
                    # pad -1 for indice ignore
                    gt_inds_arange_pad = F.pad(
                        gt_inds_arange_pad, (1, 0), mode='constant', value=-1)
                    # convert to 0~gt_num+2 for indices
                    gt_inds_arange_pad += 1
                    # now 0 is bg, >1 is fg in batch_gt_indis
                    batch_gt_indis[pred_per_cls] = gt_inds_arange_pad[
                        cur_assign_res.gt_inds + 1] - 1
                    batch_max_overlaps[
                        pred_per_cls] = cur_assign_res.max_overlaps
                    batch_gt_labels[pred_per_cls] = cur_assign_res.labels

                assign_result = AssignResult(batch_num_gts, batch_gt_indis,
                                             batch_max_overlaps,
                                             batch_gt_labels)
            else:  # for single class
                assign_result = self.bbox_assigner.assign(
                    cur_boxes.tensor,
                    cur_gt_bboxes.tensor,
                    gt_labels=cur_gt_labels)
            # sample boxes
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       cur_boxes.tensor,
                                                       cur_gt_bboxes.tensor,
                                                       cur_gt_labels)
            sampling_results.append(sampling_result)
        return sampling_results
