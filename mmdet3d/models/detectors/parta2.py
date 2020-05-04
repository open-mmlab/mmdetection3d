import torch
import torch.nn.functional as F

from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS, TwoStageDetector
from .. import builder


@DETECTORS.register_module()
class PartA2(TwoStageDetector):

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PartA2, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

    def extract_feat(self, points, img_meta):
        voxels, num_points, coors = self.voxelize(points)
        voxel_dict = dict(voxels=voxels, num_points=num_points, coors=coors)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        feats_dict = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(feats_dict['spatial_features'])
        if self.with_neck:
            neck_feats = self.neck(x)
            feats_dict.update({'neck_feats': neck_feats})
        return feats_dict, voxel_dict

    @torch.no_grad()
    def voxelize(self, points):
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_meta,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None,
                      proposals=None):
        # TODO: complete it
        feats_dict, voxels_dict = self.extract_feat(points, img_meta)

        losses = dict()

        if self.with_rpn:
            rpn_outs = self.rpn_head(feats_dict['neck_feats'])
            rpn_loss_inputs = rpn_outs + (gt_bboxes_3d, gt_labels_3d, img_meta)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals  # noqa: F841

        return losses

    def forward_test(self, **kwargs):
        return self.simple_test(**kwargs)

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def simple_test(self,
                    points,
                    img_meta,
                    gt_bboxes_3d=None,
                    proposals=None,
                    rescale=False):
        feats_dict, voxels_dict = self.extract_feat(points, img_meta)
        # TODO: complete it
        if proposals is None:
            proposal_list = self.simple_test_rpn(feats_dict['neck_feats'],
                                                 img_meta, self.test_cfg.rpn)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            feats_dict, proposal_list, img_meta, rescale=rescale)
