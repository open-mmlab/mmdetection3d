# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, List

import torch
from mmcv.ops import Voxelization, furthest_point_sample
from mmdet.models import build_detector
from mmdet.models.utils import multi_apply
from scipy.sparse.csgraph import connected_components
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models.layers.sst import scatter_v2, get_inner_win_inds
from mmdet3d.models.segmentors.base import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from .single_stage import SingleStage3DDetector
from .. import builder
from ..builder import build_backbone, build_head, build_neck
from ...structures.det3d_data_sample import OptSampleList
from ...utils.typing import SampleList

try:
    from torchex import connected_components as cc_gpu
except ImportError:
    cc_gpu = None

def fps(points, N):
    idx = furthest_point_sample(points.unsqueeze(0), N)
    idx = idx.squeeze(0).long()
    points = points[idx]
    return points

def filter_almost_empty(coors, min_points):
    new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    cnt_per_point = unq_cnt[unq_inv]
    valid_mask = cnt_per_point >= min_points
    return valid_mask

def find_connected_componets_gpu(points, batch_idx, dist):

    assert len(points) > 0
    assert cc_gpu is not None
    components_inds = cc_gpu(points, batch_idx, dist, 100, 2, False)
    assert len(torch.unique(components_inds)) == components_inds.max().item() + 1
    return components_inds

def find_connected_componets(points, batch_idx, dist):

    device = points.device
    bsz = batch_idx.max().item() + 1
    base = 0
    components_inds = torch.zeros_like(batch_idx) - 1

    for i in range(bsz):
        batch_mask = batch_idx == i
        if batch_mask.any():
            this_points = points[batch_mask]
            dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
            dist_mat = (dist_mat ** 2).sum(2) ** 0.5
            adj_mat = dist_mat < dist
            adj_mat = adj_mat.cpu().numpy()
            c_inds = connected_components(adj_mat, directed=False)[1]
            c_inds = torch.from_numpy(c_inds).to(device).int() + base
            base = c_inds.max().item() + 1
            components_inds[batch_mask] = c_inds

    assert len(torch.unique(components_inds)) == components_inds.max().item() + 1

    return components_inds

def find_connected_componets_single_batch(points, batch_idx, dist):

    device = points.device

    this_points = points
    dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
    dist_mat = (dist_mat ** 2).sum(2) ** 0.5
    # dist_mat = torch.cdist(this_points[:, :2], this_points[:, :2], p=2)
    adj_mat = dist_mat < dist
    adj_mat = adj_mat.cpu().numpy()
    c_inds = connected_components(adj_mat, directed=False)[1]
    c_inds = torch.from_numpy(c_inds).to(device).int()

    return c_inds

def ssg(points, batch_idx, num_fps, radius):
    device = points.device
    bsz = batch_idx.max().item() + 1
    base = 0
    components_inds = torch.zeros_like(batch_idx) - 2
    for i in range(bsz):
        batch_mask = batch_idx == i
        if batch_mask.any():
            this_points = points[batch_mask]
            this_inds = ssg_single_sample(this_points, num_fps, radius)
            this_inds[this_inds > -1] += base # keep -1
            base = this_inds.max().item() + 1
            components_inds[batch_mask] = this_inds
    assert (components_inds > -2).all()
    return components_inds

def ssg_single_sample(points, num_fps, radius):
    """
    a little complicated
    """
    if num_fps >= len(points):
        key_points = points
    else:
        key_points = fps(points, num_fps)

    k_dist_mat = key_points[:, None, :2] - key_points[None, :, :2] 
    k_dist_mat = (k_dist_mat ** 2).sum(2) ** 0.5 #[k, k]
    dist_mask = k_dist_mat < radius * 2 + 0.01

    triangle1 = torch.arange(len(key_points))[None, :].expand(len(key_points), -1) #[[0,1,2], [0, 1, 2]]
    triangle2 = triangle1.T #[[0, 0, 0], [1, 1, 1]]
    triangle_mask = triangle1 <= triangle2 
    dist_mask[triangle_mask] = False
    invalid_keypoints_mask = dist_mask.any(0)

    key_points = key_points[~invalid_keypoints_mask]

    dist_mat = key_points[:, None, :2] - points[None, :, :2] #[K, N]
    dist_mat = (dist_mat ** 2).sum(2) ** 0.5

    in_radius_mask = dist_mat < radius

    assert (in_radius_mask.sum(0) <= 1).all()

    valid_centers_mask = in_radius_mask.sum(0) == 1 # if a point falls into multiple balls or does not fall into any ball, it is invalid.
    assert valid_centers_mask.any()

    pos = torch.nonzero(in_radius_mask)
    cluster_inds = pos[:, 0]

    col_inds = pos[:, 1]
    sorted_col_inds, order = torch.sort(col_inds)
    cluster_inds = cluster_inds[order]
    assert (sorted_col_inds == torch.nonzero(valid_centers_mask).reshape(-1)).all()

    cluster_inds_full = cluster_inds.new_zeros(len(points)) - 1

    cluster_inds_full[valid_centers_mask] = cluster_inds

    return cluster_inds_full


def modify_cluster_by_class(cluster_inds_list):
    new_list = []
    for i, inds in enumerate(cluster_inds_list):
        cls_pad = inds.new_ones((len(inds),)) * i
        inds = torch.cat([cls_pad[:, None], inds], 1)
        # inds = F.pad(inds, (1, 0), 'constant', i)
        new_list.append(inds)
    return new_list


@MODELS.register_module()
class VoteSegmentor(Base3DSegmentor):

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 segmentation_head,
                 decode_neck=None,
                 auxiliary_head=None,
                 voxel_downsampling_size=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None,
                 tanh_dims=None,
                 **extra_kwargs):
        super().__init__(init_cfg=init_cfg)

        self.voxel_layer = Voxelization(**voxel_layer)

        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.backbone = build_backbone(backbone)
        self.segmentation_head = build_head(segmentation_head)
        self.segmentation_head.train_cfg = train_cfg
        self.segmentation_head.test_cfg = test_cfg
        self.decode_neck = build_neck(decode_neck)

        assert voxel_encoder['type'] == 'DynamicScatterVFE'


        self.print_info = {}
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cfg = train_cfg if train_cfg is not None else test_cfg
        self.num_classes = segmentation_head['num_classes']
        self.save_list = []
        self.point_cloud_range = voxel_layer['point_cloud_range']
        self.voxel_size = voxel_layer['voxel_size']
        self.voxel_downsampling_size = voxel_downsampling_size
        self.tanh_dims = tanh_dims
    
    def encode_decode(self):
        return None

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        return NotImplementedError

    @torch.no_grad()
    # @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        batch_points, coors = self.voxelize(points)
        coors = coors.long()
        voxel_features, voxel_coors, voxel2point_inds = self.voxel_encoder(batch_points, coors, return_inv=True)
        voxel_info = self.middle_encoder(voxel_features, voxel_coors)
        x = self.backbone(voxel_info)[0]
        padding = -1
        # voxel_coors_dropped = x['voxel_feats'] # bug, leave it for feature modification
        if 'shuffle_inds' not in voxel_info:
            voxel_feats_reorder = x['voxel_feats']
        else:
            # this branch only used in SST-based FSD 
            voxel_feats_reorder = self.reorder(x['voxel_feats'], voxel_info['shuffle_inds'], voxel_info['voxel_keep_inds'], padding) #'not consistent with voxel_coors any more'

        out = self.decode_neck(batch_points, coors, voxel_feats_reorder, voxel2point_inds, padding)

        return out, coors, batch_points

    def reorder(self, data, shuffle_inds, keep_inds, padding=-1):
        '''
        Padding dropped voxel and reorder voxels.  voxel length and order will be consistent with the output of voxel_encoder.
        '''
        num_voxel_no_drop = len(shuffle_inds)
        data_dim = data.size(1)

        temp_data = padding * data.new_ones((num_voxel_no_drop, data_dim))
        out_data = padding * data.new_ones((num_voxel_no_drop, data_dim))

        temp_data[keep_inds] = data
        out_data[shuffle_inds] = temp_data

        return out_data

    def voxel_downsample(self, points_list):
        device = points_list[0].device
        out_points_list = []
        voxel_size = torch.tensor(self.voxel_downsampling_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)

        for points in points_list:
            coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
            out_points, new_coors = scatter_v2(points, coors, mode='avg', return_inv=False)
            out_points_list.append(out_points)
        return out_points_list

    def loss(self,
             points,
             img_metas,
             gt_bboxes_3d,
             gt_labels_3d,
             as_subsegmentor=False,
             ):
        if self.tanh_dims is not None:
            for p in points:
                p[:, self.tanh_dims] = torch.tanh(p[:, self.tanh_dims])
        elif points[0].size(1) in (4,5):
            # a hack way to scale the intensity and elongation in WOD
            points = [torch.cat([p[:, :3], torch.tanh(p[:, 3:])], dim=1) for p in points]
        
        if self.voxel_downsampling_size is not None:
            points = self.voxel_downsample(points)

        labels, vote_targets, vote_mask = self.segmentation_head.get_targets(points, gt_bboxes_3d, gt_labels_3d)

        neck_out, pts_coors, points = self.extract_feat(points, img_metas)

        losses = dict()

        feats = neck_out[0]
        valid_pts_mask = neck_out[1]
        points = points[valid_pts_mask]
        pts_coors = pts_coors[valid_pts_mask]
        labels = labels[valid_pts_mask]
        vote_targets = vote_targets[valid_pts_mask]
        vote_mask = vote_mask[valid_pts_mask]

        assert feats.size(0) == labels.size(0)

        if as_subsegmentor:
            loss_decode, preds_dict = self.segmentation_head.loss(feats, img_metas, labels, vote_targets, vote_mask, return_preds=True)
            losses.update(loss_decode)

            seg_logits = preds_dict['seg_logits']
            vote_preds = preds_dict['vote_preds']

            offsets = self.segmentation_head.decode_vote_targets(vote_preds)

            output_dict = dict(
                seg_points=points,
                seg_logits=preds_dict['seg_logits'],
                seg_vote_preds=preds_dict['vote_preds'],
                offsets=offsets,
                seg_feats=feats,
                batch_idx=pts_coors[:, 0],
                losses=losses
            )
        else:
            loss_decode = self.segmentation_head.loss(feats, img_metas, labels, vote_targets, vote_mask, return_preds=False)
            losses.update(loss_decode)
            output_dict = losses

        return output_dict

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:

        points = batch_inputs['points']
        img_metas = batch_data_samples[0].metainfo

        if self.tanh_dims is not None:
            for p in points:
                p[:, self.tanh_dims] = torch.tanh(p[:, self.tanh_dims])
        elif points[0].size(1) in (4,5):
            points = [torch.cat([p[:, :3], torch.tanh(p[:, 3:])], dim=1) for p in points]

        if self.voxel_downsampling_size is not None:
            points = self.voxel_downsample(points)

        # seg_pred = []
        x, pts_coors, points = self.extract_feat(points, img_metas)
        feats = x[0]
        valid_pts_mask = x[1]
        points = points[valid_pts_mask]
        pts_coors = pts_coors[valid_pts_mask]

        seg_logits, vote_preds = self.segmentation_head.forward(feats)

        offsets = self.segmentation_head.decode_vote_targets(vote_preds)

        output_dict = dict(
            seg_points=points,
            seg_logits=seg_logits,
            seg_vote_preds=vote_preds,
            offsets=offsets,
            seg_feats=feats,
            batch_idx=pts_coors[:, 0],
        )

        return output_dict

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        raise NotImplementedError


@MODELS.register_module()
class SingleStageFSD(SingleStage3DDetector):

    def __init__(self,
                 backbone,
                 segmentor,
                 voxel_layer=None,
                 voxel_encoder=None,
                 middle_encoder=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 cluster_assigner=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(SingleStageFSD, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        if voxel_layer is not None:
            self.voxel_layer = Voxelization(**voxel_layer)
        if voxel_encoder is not None:
            self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        if middle_encoder is not None:
            self.middle_encoder = builder.build_middle_encoder(middle_encoder)

        self.segmentor = build_detector(segmentor)
        self.head_type = bbox_head['type']
        self.num_classes = bbox_head['num_classes']

        self.cfg = self.train_cfg if self.train_cfg else self.test_cfg
        if 'radius' in cluster_assigner:
            self.cluster_assigner = SSGAssigner(**cluster_assigner)
        elif 'hybrid' in cluster_assigner:
            cluster_assigner.pop('hybrid')
            self.cluster_assigner = HybridAssigner(**cluster_assigner)
        else:
            self.cluster_assigner = ClusterAssigner(**cluster_assigner)
        self.cluster_assigner.num_classes = self.num_classes
        self.print_info = {}
        self.as_rpn = bbox_head.get('as_rpn', False)

    @torch.no_grad()
    # @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        """
        raise ValueError('This function should not be called in FSD')
        device = points[0].device
        voxel_size = torch.tensor(self.voxel_layer.voxel_size, device=device)
        pc_range = torch.tensor(self.voxel_layer.point_cloud_range, device=device)

        coors = []
        for res in points:
            res_coors = torch.div(res[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
            res_coors = res_coors[:, [2, 1, 0]] # to zyx order
            coors.append(res_coors)

        points = torch.cat(points, dim=0)

        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)

        return points, coors_batch
        
    def extract_feat(self, points, pts_feats, pts_cluster_inds, img_metas, center_preds):
        """Extract features from points."""
        cluster_xyz, _, inv_inds = scatter_v2(center_preds, pts_cluster_inds, mode='avg', return_inv=True)

        f_cluster = points[:, :3] - cluster_xyz[inv_inds]

        out_pts_feats, cluster_feats, out_coors = self.backbone(points, pts_feats, pts_cluster_inds, f_cluster)
        out_dict = dict(
            cluster_feats=cluster_feats,
            cluster_xyz=cluster_xyz,
            cluster_inds=out_coors
        )
        if self.as_rpn:
            out_dict['cluster_pts_feats'] = out_pts_feats
            out_dict['cluster_pts_xyz'] = points

        return out_dict
    #
    # def loss(self,
    #          points,
    #          img_metas,
    #          gt_bboxes_3d,
    #          gt_labels_3d,
    #          gt_bboxes_ignore=None,
    #          runtime_info=None):

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        self.runtime_info = runtime_info # stupid way to get arguements from children class
        losses = {}
        gt_bboxes_3d = [b[l>=0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        gt_labels_3d = [l[l>=0] for l in gt_labels_3d]

        seg_out_dict = self.segmentor(points=points, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, as_subsegmentor=True)

        seg_feats = seg_out_dict['seg_feats']
        if self.train_cfg.get('detach_segmentor', False):
            seg_feats = seg_feats.detach()
        seg_loss = seg_out_dict['losses']
        losses.update(seg_loss)

        dict_to_sample = dict(
            seg_points=seg_out_dict['seg_points'],
            seg_logits=seg_out_dict['seg_logits'].detach(),
            seg_vote_preds=seg_out_dict['seg_vote_preds'].detach(),
            seg_feats=seg_feats,
            batch_idx=seg_out_dict['batch_idx'],
            vote_offsets=seg_out_dict['offsets'].detach(),
        )
        if self.cfg.get('pre_voxelization_size', None) is not None:
            dict_to_sample = self.pre_voxelize(dict_to_sample)
        sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets'], gt_bboxes_3d, gt_labels_3d) # per cls list in sampled_out

        # we filter almost empty voxel in clustering, so here is a valid_mask
        cluster_inds_list, valid_mask_list = self.cluster_assigner(sampled_out['center_preds'], sampled_out['batch_idx'], gt_bboxes_3d, gt_labels_3d,  origin_points=sampled_out['seg_points']) # per cls list
        pts_cluster_inds = torch.cat(cluster_inds_list, dim=0) #[N, 3], (cls_id, batch_idx, cluster_id)

        num_clusters = len(torch.unique(pts_cluster_inds, dim=0)) * torch.ones((1,), device=pts_cluster_inds.device).float()
        losses['num_clusters'] = num_clusters

        sampled_out = self.update_sample_results_by_mask(sampled_out, valid_mask_list)

        combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds'])

        points = combined_out['seg_points']
        pts_feats = torch.cat([combined_out['seg_logits'], combined_out['seg_vote_preds'], combined_out['seg_feats']], dim=1)
        assert len(pts_cluster_inds) == len(points) == len(pts_feats)
        losses['num_fg_points'] = torch.ones((1,), device=points.device).float() * len(points)

        extracted_outs = self.extract_feat(points, pts_feats, pts_cluster_inds, img_metas, combined_out['center_preds'])
        cluster_feats = extracted_outs['cluster_feats']
        cluster_xyz = extracted_outs['cluster_xyz']
        cluster_inds = extracted_outs['cluster_inds'] # [class, batch, groups]

        assert (cluster_inds[:, 0]).max().item() < self.num_classes

        outs = self.bbox_head(cluster_feats, cluster_xyz, cluster_inds)
        loss_inputs = (outs['cls_logits'], outs['reg_preds']) + (cluster_xyz, cluster_inds) + (gt_bboxes_3d, gt_labels_3d, img_metas)
        det_loss = self.bbox_head.loss(
            *loss_inputs, iou_logits=outs.get('iou_logits', None), gt_bboxes_ignore=gt_bboxes_ignore)
        
        if hasattr(self.bbox_head, 'print_info'):
            self.print_info.update(self.bbox_head.print_info)
        losses.update(det_loss)
        losses.update(self.print_info)

        if self.as_rpn:
            output_dict = dict(
                rpn_losses=losses,
                cls_logits=outs['cls_logits'],
                reg_preds=outs['reg_preds'],
                cluster_xyz=cluster_xyz,
                cluster_inds=cluster_inds,
                all_input_points=dict_to_sample['seg_points'],
                valid_pts_feats=extracted_outs['cluster_pts_feats'],
                valid_pts_xyz=extracted_outs['cluster_pts_xyz'],
                seg_feats=dict_to_sample['seg_feats'],
                pts_mask=sampled_out['fg_mask_list'],
                pts_batch_inds=dict_to_sample['batch_idx'],
            )
            return output_dict
        else:
            return losses

    def update_sample_results_by_mask(self, sampled_out, valid_mask_list):
        for k in sampled_out:
            old_data = sampled_out[k]
            if len(old_data[0]) == len(valid_mask_list[0]) or 'fg_mask' in k:
                if 'fg_mask' in k:
                    new_data_list = []
                    for data, mask in zip(old_data, valid_mask_list):
                        new_data = data.clone()
                        new_data[data] = mask
                        assert new_data.sum() == mask.sum()
                        new_data_list.append(new_data)
                    sampled_out[k] = new_data_list
                else:
                    new_data_list = [data[mask] for data, mask in zip(old_data, valid_mask_list)]
                    sampled_out[k] = new_data_list
        return sampled_out
    
    def combine_classes(self, data_dict, name_list):
        out_dict = {}
        for name in data_dict:
            if name in name_list:
                out_dict[name] = torch.cat(data_dict[name], 0)
        return out_dict
    
    def pre_voxelize(self, data_dict):
        batch_idx = data_dict['batch_idx']
        points = data_dict['seg_points']

        voxel_size = torch.tensor(self.cfg.pre_voxelization_size, device=batch_idx.device)
        pc_range = torch.tensor(self.cluster_assigner.point_cloud_range, device=points.device)
        coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)

        new_coors, unq_inv  = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)

        voxelized_data_dict = {}
        for data_name in data_dict:
            data = data_dict[data_name]
            if data.dtype in (torch.float, torch.float16):
                voxelized_data, voxel_coors = scatter_v2(data, coors, mode='avg', return_inv=False, new_coors=new_coors, unq_inv=unq_inv)
                voxelized_data_dict[data_name] = voxelized_data

        voxelized_data_dict['batch_idx'] = voxel_coors[:, 0]
        return voxelized_data_dict

    # def simple_test(self, points, img_metas, imgs=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        seg_out_dict = self.segmentor.predict(batch_inputs, batch_data_samples)

        seg_feats = seg_out_dict['seg_feats']

        dict_to_sample = dict(
            seg_points=seg_out_dict['seg_points'],
            seg_logits=seg_out_dict['seg_logits'],
            seg_vote_preds=seg_out_dict['seg_vote_preds'],
            seg_feats=seg_feats,
            batch_idx=seg_out_dict['batch_idx'],
            vote_offsets = seg_out_dict['offsets']
        )

        if self.cfg.get('pre_voxelization_size', None) is not None:
            dict_to_sample = self.pre_voxelize(dict_to_sample)

        sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets'])  # per cls list in sampled_out

        # we filter almost empty voxel in clustering, so here is a valid_mask
        cluster_inds_list, valid_mask_list = self.cluster_assigner(sampled_out['center_preds'], sampled_out['batch_idx'], origin_points=sampled_out['seg_points']) # per cls list

        pts_cluster_inds = torch.cat(cluster_inds_list, dim=0) # [N, 3], (cls_id, batch_idx, cluster_id)

        sampled_out = self.update_sample_results_by_mask(sampled_out, valid_mask_list)

        combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds'])

        points = combined_out['seg_points']
        pts_feats = torch.cat([combined_out['seg_logits'], combined_out['seg_vote_preds'], combined_out['seg_feats']], dim=1)
        assert len(pts_cluster_inds) == len(points) == len(pts_feats)

        extracted_outs = self.extract_feat(points, pts_feats, pts_cluster_inds, batch_data_samples, combined_out['center_preds'])
        cluster_feats = extracted_outs['cluster_feats']
        cluster_xyz = extracted_outs['cluster_xyz']
        cluster_inds = extracted_outs['cluster_inds']
        assert (cluster_inds[:, 1] == 0).all()

        outs = self.bbox_head(cluster_feats, cluster_xyz, cluster_inds)

        bbox_list = self.bbox_head.get_bboxes(
            outs['cls_logits'], outs['reg_preds'],
            cluster_xyz, cluster_inds, batch_data_samples,
            rescale=False,
            iou_logits=outs.get('iou_logits', None))

        if self.as_rpn:
            output_dict = dict(
                all_input_points=dict_to_sample['seg_points'],
                valid_pts_feats=extracted_outs['cluster_pts_feats'],
                valid_pts_xyz=extracted_outs['cluster_pts_xyz'],
                seg_feats=dict_to_sample['seg_feats'],
                pts_mask=sampled_out['fg_mask_list'],
                pts_batch_inds=dict_to_sample['batch_idx'],
                proposal_list=bbox_list
            )
            return output_dict
        else:
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            return bbox_results

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        raise NotImplementedError

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        return NotImplementedError

    
    def sample(self, dict_to_sample, offset, gt_bboxes_3d=None, gt_labels_3d=None):

        if self.cfg.get('group_sample', False):
            return self.group_sample(dict_to_sample, offset)

        # cfg = self.train_cfg if self.training else self.test_cfg

        seg_logits = dict_to_sample['seg_logits']
        assert (seg_logits < 0).any()  # make sure no sigmoid applied

        if seg_logits.size(1) == self.num_classes:
            seg_scores = seg_logits.sigmoid()
        else:
            raise NotImplementedError

        offset = offset.reshape(-1, self.num_classes, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]
        fg_mask_list = [] # fg_mask of each cls
        center_preds_list = [] # fg_mask of each cls

        batch_idx = dict_to_sample['batch_idx']
        batch_size = batch_idx.max().item() + 1
        for cls in range(self.num_classes):
            fg_mask = self.get_fg_mask(seg_scores, cls)

            if len(torch.unique(batch_idx[fg_mask])) < batch_size:
                one_random_pos_per_sample = self.get_sample_beg_position(batch_idx, fg_mask)
                fg_mask[one_random_pos_per_sample] = True # at least one point per sample

            fg_mask_list.append(fg_mask)

            this_offset = offset[fg_mask, cls, :]
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)


        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list
        output_dict['center_preds'] = center_preds_list

        return output_dict

    def get_sample_beg_position(self, batch_idx, fg_mask):
        assert batch_idx.shape == fg_mask.shape
        inner_inds = get_inner_win_inds(batch_idx.contiguous())
        pos = torch.where(inner_inds == 0)[0]
        return pos
    
    def get_fg_mask(self, seg_scores, cls_id):
        if self.training and self.train_cfg.get('disable_pretrain', False) and not self.runtime_info.get('enable_detection', False):
            seg_scores = seg_scores[:, cls_id]
            topks = self.train_cfg.get('disable_pretrain_topks', [100, 100, 100])
            k = min(topks[cls_id], len(seg_scores))
            top_inds = torch.topk(seg_scores, k)[1]
            fg_mask = torch.zeros_like(seg_scores, dtype=torch.bool)
            fg_mask[top_inds] = True
        else:
            seg_scores = seg_scores[:, cls_id]
            cls_score_thr = self.cfg['score_thresh'][cls_id]
            if self.training:
                buffer_thr = self.runtime_info.get('threshold_buffer', 0)
            else:
                buffer_thr = 0
            fg_mask = seg_scores > cls_score_thr + buffer_thr

        return fg_mask

    def split_by_batch(self, data, batch_idx, batch_size):
        assert batch_idx.max().item() + 1 <= batch_size
        data_list = []
        for i in range(batch_size):
            sample_mask = batch_idx == i
            data_list.append(data[sample_mask])
        return data_list

    def combine_by_batch(self, data_list, batch_idx, batch_size):
        assert len(data_list) == batch_size
        if data_list[0] is None:
            return None
        data_shape = (len(batch_idx),) + data_list[0].shape[1:]
        full_data = data_list[0].new_zeros(data_shape)
        for i, data in enumerate(data_list):
            sample_mask = batch_idx == i
            full_data[sample_mask] = data
        return full_data

    def group_sample(self, dict_to_sample, offset):

        """
        For argoverse 2 dataset, where the number of classes is large
        """

        bsz = dict_to_sample['batch_idx'].max().item() + 1
        assert bsz == 1, "Maybe some codes need to be modified if bsz > 1"
        # combine all classes as fg class.
        cfg = self.train_cfg if self.training else self.test_cfg

        seg_logits = dict_to_sample['seg_logits']
        assert (seg_logits < 0).any() # make sure no sigmoid applied

        assert seg_logits.size(1) == self.num_classes + 1 # we have background class
        seg_scores = seg_logits.softmax(1)

        offset = offset.reshape(-1, self.num_classes + 1, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]
        fg_mask_list = [] # fg_mask of each cls
        center_preds_list = [] # fg_mask of each cls


        cls_score_thrs = cfg['score_thresh']
        group_lens = cfg['group_lens']
        num_groups = len(group_lens)
        assert num_groups == len(cls_score_thrs)
        assert isinstance(cls_score_thrs, (list, tuple))
        grouped_score = self.gather_group(seg_scores[:, :-1], group_lens) # without background score

        beg = 0
        for i, group_len in enumerate(group_lens):
            end = beg + group_len

            fg_mask = grouped_score[:, i] > cls_score_thrs[i]

            if not fg_mask.any():
                fg_mask[0] = True # at least one point

            fg_mask_list.append(fg_mask)

            this_offset = offset[fg_mask, beg:end, :] 
            offset_weight = self.get_offset_weight(seg_logits[fg_mask, beg:end])
            assert torch.isclose(offset_weight.sum(1), offset_weight.new_ones(len(offset_weight))).all()
            this_offset = (this_offset * offset_weight[:, :, None]).sum(dim=1)
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)
            beg = end
        assert end == 26, 'for 26class argo'


        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list
        output_dict['center_preds'] = center_preds_list

        return output_dict
    
    def get_offset_weight(self, seg_logit):
        mode = self.cfg['offset_weight']
        if mode == 'max':
            weight = ((seg_logit - seg_logit.max(1)[0][:, None]).abs() < 1e-6).float()
            assert ((weight == 1).any(1)).all()
            weight = weight / weight.sum(1)[:, None] # in case of two max values
            return weight
        else:
            raise NotImplementedError
    
    def gather_group(self, scores, group_lens):
        assert (scores >= 0).all()
        score_per_group = []
        beg = 0
        for group_len in group_lens:
            end = beg + group_len
            score_this_g = scores[:, beg:end].sum(1)
            score_per_group.append(score_this_g)
            beg = end
        assert end == scores.size(1) == sum(group_lens)
        gathered_score = torch.stack(score_per_group, dim=1)
        assert gathered_score.size(1) == len(group_lens)
        return  gathered_score


class ClusterAssigner(torch.nn.Module):
    ''' Generating cluster centers for each class and assign each point to cluster centers
    '''

    def __init__(
        self,
        cluster_voxel_size,
        min_points,
        point_cloud_range,
        connected_dist,
        class_names=['Car', 'Cyclist', 'Pedestrian'],
        gpu_clustering=(False, False),
    ):
        super().__init__()
        self.cluster_voxel_size = cluster_voxel_size
        self.min_points = min_points
        self.connected_dist = connected_dist
        self.point_cloud_range = point_cloud_range
        self.class_names = class_names
        self.gpu_clustering = gpu_clustering

    @torch.no_grad()
    def forward(self, points_list, batch_idx_list, gt_bboxes_3d=None, gt_labels_3d=None, origin_points=None):
        gt_bboxes_3d = None 
        gt_labels_3d = None
        assert self.num_classes == len(self.class_names)
        cluster_inds_list, valid_mask_list = \
            multi_apply(self.forward_single_class, points_list, batch_idx_list, self.class_names, origin_points)
        cluster_inds_list = modify_cluster_by_class(cluster_inds_list)
        return cluster_inds_list, valid_mask_list

    def forward_single_class(self, points, batch_idx, class_name, origin_points):
        batch_idx = batch_idx.int()

        if isinstance(self.cluster_voxel_size, dict):
            cluster_vsize = self.cluster_voxel_size[class_name]
        elif isinstance(self.cluster_voxel_size, list):
            cluster_vsize = self.cluster_voxel_size[self.class_names.index(class_name)]
        else:
            cluster_vsize = self.cluster_voxel_size

        voxel_size = torch.tensor(cluster_vsize, device=points.device)
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').int()
        # coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)

        valid_mask = filter_almost_empty(coors, min_points=self.min_points)
        if not valid_mask.any():
            valid_mask = ~valid_mask
            # return coors.new_zeros((3,0)), valid_mask

        points = points[valid_mask]
        batch_idx = batch_idx[valid_mask]
        coors = coors[valid_mask]
        # elif len(points) 

        sampled_centers, voxel_coors, inv_inds = scatter_v2(points, coors, mode='avg', return_inv=True)

        if isinstance(self.connected_dist, dict):
            dist = self.connected_dist[class_name]
        elif isinstance(self.connected_dist, list):
            dist = self.connected_dist[self.class_names.index(class_name)]
        else:
            dist = self.connected_dist

        if self.training:
            cluster_inds = find_connected_componets(sampled_centers, voxel_coors[:, 0], dist)
        else:
            if self.gpu_clustering[1]:
                cluster_inds = find_connected_componets_gpu(sampled_centers, voxel_coors[:, 0], dist)
            else:
                cluster_inds = find_connected_componets_single_batch(sampled_centers, voxel_coors[:, 0], dist)
        assert len(cluster_inds) == len(sampled_centers)

        cluster_inds_per_point = cluster_inds[inv_inds]
        cluster_inds_per_point = torch.stack([batch_idx, cluster_inds_per_point], 1)
        return cluster_inds_per_point, valid_mask
    

class SSGAssigner(torch.nn.Module):
    ''' Generating cluster centers for each class and assign each point to cluster centers
    '''

    def __init__(
        self,
        cluster_voxel_size,
        point_cloud_range,
        radius,
        num_fps,
        class_names=['Car', 'Cyclist', 'Pedestrian'],
    ):
        super().__init__()
        self.cluster_voxel_size = cluster_voxel_size
        self.radius = radius
        self.num_fps = num_fps
        self.point_cloud_range = point_cloud_range
        self.class_names = class_names

    @torch.no_grad()
    def forward(self, points_list, batch_idx_list, gt_bboxes_3d=None, gt_labels_3d=None, origin_points=None):
        gt_bboxes_3d = None 
        gt_labels_3d = None
        assert self.num_classes == len(self.class_names)
        cluster_inds_list, valid_mask_list = \
            multi_apply(self.forward_single_class, points_list, batch_idx_list, self.class_names, origin_points)
        cluster_inds_list = modify_cluster_by_class(cluster_inds_list)
        return cluster_inds_list, valid_mask_list

    def forward_single_class(self, points, batch_idx, class_name, origin_points):

        if isinstance(self.cluster_voxel_size, dict):
            cluster_vsize = self.cluster_voxel_size[class_name]
        elif isinstance(self.cluster_voxel_size, list):
            cluster_vsize = self.cluster_voxel_size[self.class_names.index(class_name)]
        else:
            cluster_vsize = self.cluster_voxel_size

        if isinstance(self.radius, dict):
            radius = self.radius[class_name]
        elif isinstance(self.radius, list):
            radius = self.radius[self.class_names.index(class_name)]
        else:
            radius = self.radius

        voxel_size = torch.tensor(cluster_vsize, device=points.device)
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)

        voxels, _, inv_inds = scatter_v2(points, coors, mode='avg', return_inv=True)

        num_fps = self.num_fps[class_name]
        if num_fps >= len(voxels):
            key_points = voxels
        else:
            key_points = fps(voxels, self.num_fps[class_name])

        k_dist_mat = key_points[:, None, :2] - key_points[None, :, :2] 
        k_dist_mat = (k_dist_mat ** 2).sum(2) ** 0.5 #[k, k]
        dist_mask = k_dist_mat < radius * 2 + 0.01

        triangle1 = torch.arange(len(key_points))[None, :].expand(len(key_points), -1) #[[0,1,2], [0, 1, 2]]
        triangle2 = triangle1.T #[[0, 0, 0], [1, 1, 1]]
        triangle_mask = triangle1 <= triangle2 
        dist_mask[triangle_mask] = False
        invalid_keypoints_mask = dist_mask.any(0)

        key_points = key_points[~invalid_keypoints_mask]

        dist_mat = key_points[:, None, :2] - voxels[None, :, :2] #[K, N]
        dist_mat = (dist_mat ** 2).sum(2) ** 0.5

        in_radius_mask = dist_mat < radius

        assert (in_radius_mask.sum(0) <= 1).all()

        valid_centers_mask = in_radius_mask.sum(0) == 1
        assert valid_centers_mask.any()

        pos = torch.nonzero(in_radius_mask)
        cluster_inds = pos[:, 0]

        col_inds = pos[:, 1]
        sorted_col_inds, order = torch.sort(col_inds)
        cluster_inds = cluster_inds[order]
        assert (sorted_col_inds == torch.nonzero(valid_centers_mask).reshape(-1)).all()

        cluster_inds_full = cluster_inds.new_zeros(len(voxels)) - 1

        cluster_inds_full[valid_centers_mask] = cluster_inds

        cluster_inds_per_point = cluster_inds_full[inv_inds]
        valid_pts_mask = cluster_inds_per_point > -1

        cluster_inds_per_point = torch.stack([batch_idx, cluster_inds_per_point], 1)
        cluster_inds_per_point = cluster_inds_per_point[valid_pts_mask]

        return cluster_inds_per_point, valid_pts_mask
    

class HybridAssigner(torch.nn.Module):
    ''' Generating cluster centers for each class and assign each point to cluster centers
    '''

    def __init__(
        self,
        point_cloud_range,
        cfg_per_class,
        class_names=['Car', 'Cyclist', 'Pedestrian'],
    ):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.class_names = class_names
        self.cfg_per_class = cfg_per_class

    @torch.no_grad()
    def forward(self, points_list, batch_idx_list, gt_bboxes_3d=None, gt_labels_3d=None, origin_points=None):
        gt_bboxes_3d = None 
        gt_labels_3d = None
        assert self.num_classes == len(self.class_names)
        cluster_inds_list, valid_mask_list = \
            multi_apply(self.forward_single_class, points_list, batch_idx_list, self.class_names, origin_points)
        cluster_inds_list = modify_cluster_by_class(cluster_inds_list)
        return cluster_inds_list, valid_mask_list
    
    def forward_single_class(self, points, batch_idx, class_name, origin_points):
        """
        Dispatcher
        """
        assigner_type = self.cfg_per_class[class_name]['assigner_type']
        if assigner_type == 'ssg':
            return self.forward_ssg(points, batch_idx, class_name, origin_points)
        elif assigner_type == 'ccl':
            return self.forward_ccl(points, batch_idx, class_name, origin_points)

    def forward_ssg(self, points, batch_idx, class_name, origin_points):

        cluster_vsize = self.cfg_per_class[class_name]['cluster_voxel_size']
        radius = self.cfg_per_class[class_name]['radius']
        num_fps = self.cfg_per_class[class_name]['num_fps']

        voxel_size = torch.tensor(cluster_vsize, device=points.device)
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)

        voxels, voxel_coors, inv_inds = scatter_v2(points, coors, mode='avg', return_inv=True)

        cluster_inds_full = ssg(voxels, voxel_coors[:, 0], num_fps, radius)

        cluster_inds_per_point = cluster_inds_full[inv_inds]
        valid_pts_mask = cluster_inds_per_point > -1

        cluster_inds_per_point = torch.stack([batch_idx, cluster_inds_per_point], 1)
        cluster_inds_per_point = cluster_inds_per_point[valid_pts_mask]

        return cluster_inds_per_point, valid_pts_mask
    

    def forward_ccl(self, points, batch_idx, class_name, origin_points):

        cluster_vsize = self.cfg_per_class[class_name]['cluster_voxel_size']
        min_points = self.cfg_per_class[class_name]['min_points']
        dist = self.cfg_per_class[class_name]['connected_dist']

        voxel_size = torch.tensor(cluster_vsize, device=points.device)
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)

        valid_mask = filter_almost_empty(coors, min_points=min_points)
        if not valid_mask.any():
            valid_mask = ~valid_mask
            # return coors.new_zeros((3,0)), valid_mask

        points = points[valid_mask]
        batch_idx = batch_idx[valid_mask]
        coors = coors[valid_mask]
        # elif len(points) 

        sampled_centers, voxel_coors, inv_inds = scatter_v2(points, coors, mode='avg', return_inv=True)


        cluster_inds = find_connected_componets(sampled_centers, voxel_coors[:, 0], dist)
        assert len(cluster_inds) == len(sampled_centers)

        cluster_inds_per_point = cluster_inds[inv_inds]
        cluster_inds_per_point = torch.stack([batch_idx, cluster_inds_per_point], 1)
        return cluster_inds_per_point, valid_mask