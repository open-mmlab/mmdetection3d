from typing import Optional, Union

from torch import nn, Tensor

from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList

from typing import Optional, Union


@MODELS.register_module()
class TPVFormerOCC(Base3DSegmentor):
    """TPVFormer: 3D Object Detection from Multi-view Images via 3D-to-2D Queries

    Args:
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
        use_grid_mask (bool) : Data augmentation. Whether to mask out some
            grids during extract_img_feat. Defaults to False.
        img_backbone (dict, optional): Backbone of extracting
            images feature. Defaults to None.
        img_neck (dict, optional): Neck of extracting
            image features. Defaults to None.
        pts_bbox_head (dict, optional): Bboxes head of
            detr3d. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
    """
    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 backbone=None,
                 neck=None,
                 encoder=None,
                 decode_head=None):

        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.encoder = MODELS.build(encoder)
        self.decode_head = MODELS.build(decode_head)

    def extract_feat(self,  batch_inputs_dict: dict) -> Tensor:

        """Extract features of images."""
        img = batch_inputs_dict
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        img_feats = self.backbone(img)

        if hasattr(self, 'neck'):
            img_feats = self.neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def _forward(self, 
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`. Defaults to None.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """   
        img_feats = self.extract_feat(batch_inputs_dict)
        outs = self.encoder(img_feats, batch_data_samples)
        outs = self.decode_head(outs, batch_inputs_dict['voxels']['coors'])

        return outs

    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> SampleList:
        img_feats = self.extract_feat(batch_inputs['imgs'])
        queries = self.encoder(img_feats, batch_data_samples)
        losses = self.decode_head.loss(queries, batch_data_samples)

        return losses

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""

        img_feats = self.extract_feat(batch_inputs['imgs'])
        tpv_queries = self.encoder(img_feats, batch_data_samples)
        seg_logits = self.decode_head.predict(tpv_queries, batch_data_samples)
        seg_preds = [seg_logit.argmax(dim=1) for seg_logit in seg_logits] # torch.Size([1, 34720, 1, 1])
        seg_preds = seg_preds[1] # TODO (chirs)  

        return self.postprocess_result(seg_preds, batch_data_samples)

    def aug_test(self, batch_inputs, batch_data_samples):
        pass

    def encode_decode(self, batch_inputs: dict,
                      batch_data_samples: SampleList) -> SampleList:
        pass


@MODELS.register_module()
class TPVFormer(Base3DSegmentor):
    """TPVFormer: 3D Object Detection from Multi-view Images via 3D-to-2D Queries

    Args:
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
        use_grid_mask (bool) : Data augmentation. Whether to mask out some
            grids during extract_img_feat. Defaults to False.
        img_backbone (dict, optional): Backbone of extracting
            images feature. Defaults to None.
        img_neck (dict, optional): Neck of extracting
            image features. Defaults to None.
        pts_bbox_head (dict, optional): Bboxes head of
            detr3d. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
    """
    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 backbone=None,
                 neck=None,
                 encoder=None,
                 decode_head=None):

        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.encoder = MODELS.build(encoder)
        self.decode_head = MODELS.build(decode_head)

    def extract_feat(self, img):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img_feats = self.backbone(img)

        if hasattr(self, 'neck'):
            img_feats = self.neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def _forward(self, 
                 batch_inputs: dict, 
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Forward training function."""
        img_feats = self.extract_feat(batch_inputs['imgs'])
        outs = self.encoder(img_feats, batch_data_samples)
        outs = self.decode_head(outs, batch_inputs['voxels']['coors'])
        return outs

    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> SampleList:
        img_feats = self.extract_feat(batch_inputs['imgs'])
        queries = self.encoder(img_feats, batch_data_samples)
        losses = self.decode_head.loss(queries, batch_data_samples)
        return losses

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""
        img_feats = self.extract_feat(batch_inputs['imgs'])
        tpv_queries = self.encoder(img_feats, batch_data_samples)
        seg_logits = self.decode_head.predict(tpv_queries, batch_data_samples)
        seg_preds = [seg_logit.argmax(dim=1) for seg_logit in seg_logits]

        return self.postprocess_result(seg_preds, batch_data_samples)

    def aug_test(self, batch_inputs, batch_data_samples):
        pass

    def encode_decode(self, batch_inputs: dict,
                      batch_data_samples: SampleList) -> SampleList:
        pass
