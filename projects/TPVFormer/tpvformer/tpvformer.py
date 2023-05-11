from typing import Optional, Union

from torch import nn

from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class TPVFormer(Base3DSegmentor):

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

    def _forward(self, batch_inputs, batch_data_samples):
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
