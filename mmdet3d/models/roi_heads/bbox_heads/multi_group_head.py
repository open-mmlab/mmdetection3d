import copy
import torch
from collections import defaultdict
from mmcv.cnn import build_conv_layer, build_norm_layer, kaiming_init
from torch import nn

from mmdet.models import FeatureAdaption
from ...builder import HEADS, build_loss


class CenterPointFeatureAdaption(FeatureAdaption):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deform conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deform_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deform_groups=4):
        super().__init__(in_channels, out_channels, kernel_size, deform_groups)
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            in_channels, deform_groups * offset_channels, 1, bias=True)

    def forward(self, x):
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x


@HEADS.register_module()
class SepHead(nn.Module):
    """SepHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
    """

    def __init__(
            self,
            in_channels,
            heads,
            head_conv=64,
            final_kernel=1,
            init_bias=-2.19,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc_layers = []
            for i in range(num_conv - 1):
                fc_layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True))
                if norm_cfg:
                    fc_layers.append(build_norm_layer(norm_cfg, head_conv)[1])
                fc_layers.append(nn.ReLU())

            fc_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            fc = nn.Sequential(*fc_layers)
            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@HEADS.register_module()
class DCNSepHead(nn.Module):
    """DCNSepHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        num_cls (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
    """

    def __init__(
            self,
            in_channels,
            num_cls,
            heads,
            head_conv=64,
            final_kernel=1,
            init_bias=-2.19,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            **kwargs,
    ):
        super(DCNSepHead, self).__init__(**kwargs)

        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = CenterPointFeatureAdaption(
            in_channels, in_channels, kernel_size=3, deform_groups=4)

        self.feature_adapt_reg = CenterPointFeatureAdaption(
            in_channels, in_channels, kernel_size=3, deform_groups=4)

        # heatmap prediction head
        self.cls_head = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                head_conv,
                kernel_size=3,
                padding=1,
                bias=True), build_norm_layer(norm_cfg, num_features=64),
            nn.ReLU(inplace=True),
            build_conv_layer(
                conv_cfg,
                head_conv,
                num_cls,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True))
        self.cls_head[-1].bias.data.fill_(init_bias)

        # other regression target
        self.task_head = SepHead(
            in_channels, heads, head_conv=head_conv, final_kernel=final_kernel)

    def forward(self, x):
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret['hm'] = cls_score

        return ret


@HEADS.register_module
class CenterHead(nn.Module):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.
        in_channels (list[int] | int): Channels of the input feature map.
            Default: [128].
        tasks (list[dict]): Task information including class number
            and class names. Default: [].
        dataset (str): Name of the dataset. Default: 'nuscenes'.
        weight (float): Weight for location loss. Default: 0.25.
        code_weights (list[int]): Code weights for location loss. Default: [].
        common_heads (dict): Conv information for common heads.
            Default: dict().
        crit (dict): Config of classification loss function.
            Default: dict(type='CenterPointFocalLoss').
        crit_reg (dict): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        init_bias (float): Initial bias. Default: -2.19.
        share_conv_channel (int): Output channels for share_conv_layer.
            Default: 64.
        num_hm_conv (int): Number of conv layers for heatmap conv layer.
            Default: 2.
        dcn_head (bool): Whether to use dcn_head. Default: False.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
    """

    def __init__(
            self,
            mode='3d',
            in_channels=[
                128,
            ],
            tasks=[],
            dataset='nuscenes',
            weight=0.25,
            code_weights=[],
            common_heads=dict(),
            crit=dict(type='CenterPointFocalLoss'),
            crit_reg=dict(type='L1Loss', reduction='none'),
            init_bias=-2.19,
            share_conv_channel=64,
            num_hm_conv=2,
            dcn_head=False,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
    ):
        super(CenterHead, self).__init__()

        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.code_weights = code_weights
        self.weight = weight  # weight between hm loss and loc loss
        self.dataset = dataset

        self.encode_background_as_zeros = True
        self.use_sigmoid_score = True
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = build_loss(crit)
        self.crit_reg = build_loss(crit_reg)
        self.loss_aux = None

        self.box_n_dim = 9  # change this if your box is different
        self.num_anchor_per_locs = [n for n in num_classes]
        self.use_direction_classifier = False

        self.bev_only = True if mode == 'bev' else False

        # a shared convolution
        self.shared_conv = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                share_conv_channel,
                kernel_size=3,
                padding=1,
                bias=True),
            build_norm_layer(norm_cfg, share_conv_channel)[1],
            nn.ReLU(inplace=True))

        self.tasks = nn.ModuleList()
        print('Use HM Bias: ', init_bias)

        if dcn_head:
            print('Use Deformable Convolution in the CenterHead!')

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            if not dcn_head:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                self.tasks.append(
                    SepHead(
                        share_conv_channel,
                        heads,
                        init_bias=init_bias,
                        final_kernel=3))

    def forward(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def loss(self, example, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            example (dict): Annos for preds.
            preds_dicts (dict): Output of forward function.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])
            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id])

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.dataset == 'nuscenes':
                preds_dict['anno_box'] = torch.cat(
                    (preds_dict['reg'], preds_dict['height'],
                     preds_dict['dim'], preds_dict['vel'], preds_dict['rot']),
                    dim=1)
            else:
                raise NotImplementedError()

            loss = 0
            ret = {}

            # Regression loss for dimension, offset, height, rotation
            ind = example['ind'][task_id]
            pred = preds_dict['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = torch.unsqueeze(example['mask'][task_id],
                                   -1).expand(pred.size())
            box_loss = torch.sum(self.crit_reg(pred, target_box, mask), 1) / (
                example['mask'][task_id].float().sum() + 1e-4)

            loc_loss = (box_loss *
                        box_loss.new_tensor(self.code_weights)).sum()

            loss += hm_loss + self.weight * loc_loss

            ret.update({
                'loss': loss,
                'hm_loss': hm_loss.detach().cpu(),
                'loc_loss': loc_loss,
                'loc_loss_elem': box_loss.detach().cpu(),
                'num_positive': example['mask'][task_id].float().sum()
            })

            rets.append(ret)
        # convert batch-key to key-batch
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged
