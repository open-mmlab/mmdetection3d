import numpy as np
import torch
from mmcv.runner import force_fp32
from torch import nn as nn

from mmdet3d.core import limit_period, xywhr2xyxyr
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.models import HEADS
from .anchor3d_head import Anchor3DHead


@HEADS.register_module()
class PointRPNHead(nn.Module):

    def __init__(self,
                 num_class,
                 input_channels,
                 model_cfg,
                 box_coder,
                 fc_config,
                 predict_boxes_when_training=False):
        super().__init__()
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class)

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(
            box_coder_utils,
            target_cfg.BOX_CODER)(**target_cfg.BOX_CODER_CONFIG)
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size)

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def forward(self, feat_dict):
        point_features = torch.concat(feat_dict['sa_features'][-1],
                                      feat_dict['sa_xyz'][-1])

        point_cls_preds = self.cls_layers(
            point_features)  # (total_points, num_class)
        point_box_preds = self.box_layers(
            point_features)  # (total_points, box_code_size)

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
            'point_box_preds': point_box_preds
        }
