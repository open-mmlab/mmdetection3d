import copy
import numpy as np
import torch
from collections import defaultdict
from mmcv.cnn import build_conv_layer, build_norm_layer, kaiming_init
from torch import nn

from mmdet3d.core import circle_nms
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models import FeatureAdaption
from ...builder import HEADS, build_loss


def gaussian2D(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius +
                 right]).to(heatmap.device).to(torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


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
        """Forward function for CenterPointFeatureAdaption.

        Args:
            x (torch.Tensor): Input feature with the shape of [B, 64, W, H].

        Returns:
            torch.Tensor: Output feature with the same shape of input feature.
        """
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
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict:   -reg （torch.Tensor): 2D regression value with the
                        shape of [B, 2, H, W].
                    -height (torch.Tensor): Height value with the
                        shape of [B, 1, H, W].
                    -dim (torch.Tensor): Size value with the shape
                        of [B, 3, H, W].
                    -rot (torch.Tensor): Rotation value with the
                        shape of [B, 2, H, W].
                    -vel (torch.Tensor): Velocity value with the
                        shape of [B, 2, H, W].
                    -hm (torch.Tensor): Heatmap with the shape of
                        [B, N, H, W].
        """
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
        cls_head = [
            build_conv_layer(
                conv_cfg,
                in_channels,
                head_conv,
                kernel_size=3,
                padding=1,
                bias=True),
            build_norm_layer(norm_cfg, num_features=64)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                conv_cfg,
                head_conv,
                num_cls,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True)
        ]
        self.cls_head = nn.Sequential(*cls_head)
        self.cls_head[-1].bias.data.fill_(init_bias)

        # other regression target
        self.task_head = SepHead(
            in_channels, heads, head_conv=head_conv, final_kernel=final_kernel)

    def forward(self, x):
        """Forward function for DCNSepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict:   -reg （torch.Tensor): 2D regression value with the
                        shape of [B, 2, H, W].
                    -height (torch.Tensor): Height value with the
                        shape of [B, 1, H, W].
                    -dim (torch.Tensor): Size value with the shape
                        of [B, 3, H, W].
                    -rot (torch.Tensor): Rotation value with the
                        shape of [B, 2, H, W].
                    -vel (torch.Tensor): Velocity value with the
                        shape of [B, 2, H, W].
                    -hm (torch.Tensor): Heatmap with the shape of
                        [B, N, H, W].
        """
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
            train_cfg=None,
            test_cfg=None,
            bbox_coder=None,
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
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encode_background_as_zeros = True
        self.use_sigmoid_score = True
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = build_loss(crit)
        self.crit_reg = build_loss(crit_reg)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_aux = None
        if dataset == 'nuscenes':
            self.box_n_dim = 9
        else:
            raise NotImplementedError
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
            else:
                self.tasks.append(
                    DCNSepHead(
                        share_conv_channel,
                        num_cls,
                        heads,
                        init_bias=init_bias,
                        final_kernel=3))

    def init_weights(self):
        pass

    def forward_single(self, x):
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

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return multi_apply(self.forward_single, feats)

    def _sigmoid(self, x):
        """Sigmoid function for input feature.

        Args:
            x (torch.Tensor): Input feature map with the shape of [B, N, H, W].

        Returns:
            torch.Tensor: Feature map after sigmoid.
        """
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            list[torch.Tensor]: Heatmap scores.
            list[torch.Tensor]: Ground truth boxes.
            list[torch.Tensor]: Indexes indicating the
                position of the valid boxes.
            list[torch.Tensor]: Masks indicating which boxes are valid.
        """
        hms, anno_boxes, inds, masks = multi_apply(self.get_targets_single,
                                                   gt_bboxes_3d, gt_labels_3d)
        hms = np.array(hms).transpose(1, 0).tolist()
        hms = [torch.stack(hms_) for hms_ in hms]
        anno_boxes = np.array(anno_boxes).transpose(1, 0).tolist()
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        inds = np.array(inds).transpose(1, 0).tolist()
        inds = [torch.stack(inds_) for inds_ in inds]
        masks = np.array(masks).transpose(1, 0).tolist()
        masks = [torch.stack(masks_) for masks_ in masks]
        return hms, anno_boxes, inds, masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            list[torch.Tensor]: Heatmap scores.
            list[torch.Tensor]: Ground truth boxes.
            list[torch.Tensor]: Indexes indicating the position
                of the valid boxes.
            list[torch.Tensor]: Masks indicating which boxes
                are valid.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d.limit_yaw(offset=0.5, period=np.pi * 2)
        gt_bboxes_3d = gt_bboxes_3d.tensor[:, [0, 1, 2, 3, 4, 5, 7, 8, 6]].to(
            device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = np.array(self.train_cfg['grid_size'])
        pc_range = np.array(self.train_cfg['point_cloud_range'])
        voxel_size = np.array(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + 1 + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).to(device))
            flag2 += len(mask)
        draw_gaussian = draw_umich_gaussian

        hms, anno_boxes, inds, masks = [], [], [], []

        for idx, task in enumerate(self.tasks):
            hm = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width, length, _ = task_boxes[idx][k][3], task_boxes[idx][k][
                    4], task_boxes[idx][k][5]
                width, length = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor'], length / voxel_size[
                        1] / self.train_cfg['out_size_factor']
                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x, coor_y = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor'], (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    ct = torch.tensor([coor_x, coor_y],
                                      dtype=torch.float32,
                                      device=device)
                    ct_int = ct.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= ct_int[0] < feature_map_size[0]
                            and 0 <= ct_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(hm[cls_id], ct, radius)

                    new_idx = k
                    x, y = ct_int[0], ct_int[1]

                    if not (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1]):
                        # a double check, should never happen
                        print(x, y, y * feature_map_size[0] + x)
                        assert False

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1

                    vx, vy = task_boxes[idx][k][6:8]
                    rot = task_boxes[idx][k][8]
                    if not self.train_cfg['no_log']:
                        anno_box[new_idx] = torch.cat([
                            ct - torch.tensor([x, y], device=device),
                            torch.unsqueeze(z, dim=0),
                            torch.log(task_boxes[idx][k][3:6]),
                            torch.unsqueeze(vx, dim=0),
                            torch.unsqueeze(vy, dim=0),
                            torch.unsqueeze(torch.sin(rot), dim=0),
                            torch.unsqueeze(torch.cos(rot), dim=0)
                        ])
                    else:
                        anno_box[new_idx] = torch.cat([
                            ct - torch.tensor([x, y], device=device),
                            torch.unsqueeze(z, dim=0), task_boxes[idx][k][3:6],
                            torch.unsqueeze(vx, dim=0),
                            torch.unsqueeze(vy, dim=0),
                            torch.unsqueeze(torch.sin(rot), dim=0),
                            torch.unsqueeze(torch.cos(rot), dim=0)
                        ],
                                                      dim=0)

            hms.append(hm)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)

        return hms, anno_boxes, inds, masks

    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.
        """
        hms, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['hm'] = self._sigmoid(preds_dict[0]['hm'])
            hm_loss = self.crit(preds_dict[0]['hm'], hms[task_id])

            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.dataset == 'nuscenes':
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['vel'],
                     preds_dict[0]['rot']),
                    dim=1)
            else:
                raise NotImplementedError()

            loss = 0
            ret = {}

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = torch.unsqueeze(masks[task_id], -1).expand(pred.size())
            box_loss = torch.sum(self.crit_reg(pred, target_box, mask), 1) / (
                masks[task_id].float().sum() + 1e-4)

            loc_loss = (box_loss *
                        box_loss.new_tensor(self.code_weights)).sum()

            loss += hm_loss + self.weight * loc_loss

            ret.update({
                'loss': loss,
                'hm_loss': hm_loss.detach().cpu(),
                'loc_loss': loc_loss,
                'loc_loss_elem': box_loss.detach().cpu(),
                'num_positive': masks[task_id].float().sum()
            })

            rets.append(ret)
        # convert batch-key to key-batch
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['hm'].shape[0]
            batch_hm = preds_dict[0]['hm'].sigmoid_()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if not self.test_cfg['no_log']:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_hm,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)

            ret_task = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                centers = boxes3d[:, [0, 1]]
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                keep = circle_nms(
                    boxes,
                    self.test_cfg['min_radius'][task_id],
                    post_max_size=self.test_cfg['post_max_size'])

                boxes3d = boxes3d[keep]
                scores = scores[keep]
                labels = labels[keep]
                ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_task.append(ret)
            rets.append(ret_task)

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = img_metas[i]['box_type_3d'](torch.cat([
                        ret[i][k] for ret in rets
                    ]), self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k] for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list
