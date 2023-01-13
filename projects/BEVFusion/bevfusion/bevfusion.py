from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from mmengine.structures import InstanceData
from torch import Tensor, nn
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList


@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        vtransform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)
        self.vtransform = MODELS.build(vtransform)
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(fusion_layer)

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['metas']))

        return outputs

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.vtransform(
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, voxel_dict['coors'],
                                    batch_size)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    # @torch.no_grad()
    # def voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders['lidar']['voxelize'](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode='constant', value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(
    #                 dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_size = len(batch_input_metas)
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        outputs = [{} for _ in range(batch_size)]
        if self.with_bbox_head:
            pred_dict = self.bbox_head.predict(feats, batch_input_metas)
            for k, (boxes, scores, labels) in enumerate(pred_dict):
                outputs[k].update({
                    'boxes_3d': boxes.to('cpu'),
                    'scores_3d': scores.cpu(),
                    'labels_3d': labels.cpu(),
                })
        if self.with_seg_head:
            gt_masks_bev = batch_input_metas.get('gt_masks_bev', None)
            logits = self.seg_head(feats, batch_input_metas)
            for k in range(batch_size):
                outputs[k].update({
                    'masks_bev': logits[k].cpu(),
                    'gt_masks_bev': gt_masks_bev[k].cpu(),
                })

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        camera2ego = batch_input_metas.get('camera2ego', None)
        lidar2ego = batch_input_metas.get('lidar2ego', None)
        lidar2camera = batch_input_metas.get('lidar2camera', None)
        lidar2image = batch_input_metas.get('lidar2image', None)
        camera_intrinsics = batch_input_metas.get('camera_intrinsics', None)
        camera2lidar = batch_input_metas.get('camera2lidar', None)
        img_aug_matrix = batch_input_metas.get('img_aug_matrix', None)
        lidar_aug_matrix = batch_input_metas.get('lidar_aug_matrix', None)
        img_feature = self.extract_img_feat(
            imgs,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            batch_input_metas,
        )
        pts_feature = self.extract_pts_feat(batch_inputs_dict)

        features = [img_feature, pts_feature]

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        pass
        # outputs = {}
        # for type, head in self.heads.items():
        #     if type == "object":
        #         pred_dict = head(x, metas)
        #         losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
        #     elif type == "map":
        #         losses = head(x, gt_masks_bev)
        #     else:
        #         raise ValueError(f"unsupported head: {type}")
        #     for name, val in losses.items():
        #         if val.requires_grad:
        #             outputs[
        #                 f"loss/{type}/{name}"] = val * self.loss_scale[type]
        #         else:
        #             outputs[f"stats/{type}/{name}"] = val
        # return outputs
