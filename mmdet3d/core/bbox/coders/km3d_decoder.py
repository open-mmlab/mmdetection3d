from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


def _nms(heat, kernel=3):
    """Process heat with Non Maximum Suppression.

        Args:
            heat (torch.Tensor): Predictions from model, batchsize * channels * h * w
            kernel (int): kernel size for max pooling

        Returns:
            torch.Tensor: batchsize * channels * h * w
    """
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _left_aggregate(heat):
    """Process heat with left aggregate.

        Args:
            heat (torch.Tensor): Predictions from model, batchsize * channels * h * w

        Returns:
            torch.Tensor: batchsize * channels * h * w
    """
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _right_aggregate(heat):
    """Process heat with right aggregate.

        Args:
            heat (torch.Tensor): Predictions from model, batchsize * channels * h * w

        Returns:
            torch.Tensor: batchsize * channels * h * w
    """
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _top_aggregate(heat):
    """Process heat with top aggregate.

        Args:
            heat (torch.Tensor): Predictions from model, batchsize * channels * h * w

        Returns:
            torch.Tensor: batchsize * channels * h * w
    """
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _bottom_aggregate(heat):
    """Process heat with bottom aggregate.

        Args:
            heat (torch.Tensor): Predictions from model, batchsize * channels * h * w

        Returns:
            torch.Tensor: batchsize * channels * h * w
    """
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _h_aggregate(heat, aggr_weight=0.1):
    """Process heat with horizontal aggregate which is composed of left and right aggregate.

        Args:
            heat (torch.Tensor): Predictions from model, batchsize * channels * h * w
            aggr_weight (float): Aggregate weights for both the left and the right aggregate

        Returns:
            torch.Tensor: batchsize * channels * h * w
    """
    return aggr_weight * _left_aggregate(heat) + \
           aggr_weight * _right_aggregate(heat) + heat


def _v_aggregate(heat, aggr_weight=0.1):
    """Process heat with vertical aggregate which is composed of top and bottom aggregate.

        Args:
            heat (torch.Tensor): Predictions from model, batchsize * channels * h * w
            aggr_weight (float): Aggregate weights for both the top and the bottom aggregate

        Returns:
            torch.Tensor: batchsize * channels * h * w
    """
    return aggr_weight * _top_aggregate(heat) + \
           aggr_weight * _bottom_aggregate(heat) + heat


def _topk_channel(scores, K=40):
    """Select top k channels of scores, index, y's, and x's given scores.

        Args:
            scores (torch.Tensor): batchsize * channels * h * w
            K (int): top K value

        Returns:
            torch.Tensor: batchsize * channels * K, top k scores
            torch.Tensor: batchsize * channels * K, top k index
            torch.Tensor: batchsize * channels * K, top k y's
            torch.Tensor: batchsize * channels * K, top k x's
    """
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _gather_feat(feat, ind, mask=None):
    """Gather feat with assigned index, with index expanded to the third dimension.

        Args:
            feat (torch.Tensor): Predictions from model, batchsize * channels * h * w
            ind (torch.Tensor): index to be gathered

        Returns:
            torch.Tensor: feat, batchsize * channels * h * w
    """

    # expand ind with the third dimension to be the same as feat
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """gather feat with assigned index, with feat be permuted with (0,2,3,1) and reshaped to three dimensions.

        Args:
            heat (torch.Tensor): Predictions from model, batchsize * channels * h * w
            ind (torch.Tensor): index to be gathered

        Returns:
            torch.Tensor: feat, batchsize * channels * h * w
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat



def _topk(scores, K=40):
    """Select top k batches of scores, index, classes, y's, and x's given scores.

        Args:
            scores (torch.Tensor): batchsize * channels * h * w
            K (int): top K value

        Returns:
            torch.Tensor: batchsize * K, top k scores
            torch.Tensor: batchsize * K, top k index
            torch.Tensor: batchsize * K, top k classes
            torch.Tensor: batchsize * K, top k y's
            torch.Tensor: batchsize * K, top k x's
    """
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def gen_position(kps, dim, rot, meta, const):
    """generate position from 2D frame to world frame.

        Args:
            kps (torch.Tensor): 1 * batchsize * 18
            dim (int): 1 * batchsize * 3 (h, w, l)
            rot (torch.Tensor): 1* batchsize * 8
            meta (dict): should contain keys below.
                - trans_output_inv: transition parameters for computing A in pinv's matrix.
                - calib: calibration for computing A in pinv's matrix.
            const (torch.Tensor): 1 * batchsize * 16 * 2

        Returns:
            torch.Tensor: 1 * batchsize * 3: pinv, coordinates in world frame
            torch.Tensor: 1 * batchsize * 1: rot_y, yaw angle
            torch.Tensor: 1 * batchsize * 18: kps, key points coordinates
    """
    b = kps.size(0)  # 1
    c = kps.size(1)  # 100
    opinv = meta['trans_output_inv']  # 1 2 3
    calib = meta['calib']  # 1 3 4

    opinv = opinv.unsqueeze(1)  # 1 1 2 3
    opinv = opinv.expand(b, c, -1, -1).contiguous().view(-1, 2, 3).float()  # 100 2 3
    kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2)  # 1 100 2 9
    hom = torch.ones(b, c, 1, 9).cuda()  # 1 100 1 9
    kps = torch.cat((kps, hom), dim=2).view(-1, 3, 9)  # 100 3 9

    kps = torch.bmm(opinv, kps).view(b, c, 2, 9)  # 1 100 2 9     [[xxxxxxx...],[yyyyyy...]]
    kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # 1 100 18 [xyxyxyxyx....]
    si = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1]  # 1 100 1
    alpha_idx = rot[:, :, 1] > rot[:, :, 5]  # 1 100
    alpha_idx = alpha_idx.float()  # 1 100
    alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi)  # 1 100
    alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)  # 1 100
    alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)  # 1 100
    alpna_pre = alpna_pre.unsqueeze(2)  # 1 100

    rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si)  # 1 100 1
    rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi  # 1 100 1
    rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi  # 1 100 1

    calib = calib.unsqueeze(1)  # 1 1 3 4
    calib = calib.expand(b, c, -1, -1).contiguous()  # 1 100 3 4
    kpoint = kps[:, :, :16]  # 1 100 16
    f = calib[:, :, 0, 0].unsqueeze(2)  # 1 100 1
    f = f.expand_as(kpoint)  # 1 100 16
    cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)  # 1 100 1 , 1 100 1
    cxy = torch.cat((cx, cy), dim=2)  # 1 100 2
    cxy = cxy.repeat(1, 1, 8)  # 1 100 16
    kp_norm = (kpoint - cxy) / f  # 1 100 16

    l = dim[:, :, 2:3]  # 1 100 1
    h = dim[:, :, 0:1]  # 1 100 1
    w = dim[:, :, 1:2]  # 1 100 1
    cosori = torch.cos(rot_y)  # 1 100 1
    sinori = torch.sin(rot_y)  # 1 100 1

    B = torch.zeros_like(kpoint)  ##1 100 16
    C = torch.zeros_like(kpoint)  # 1 100 16

    kp = kp_norm.unsqueeze(3)  # 1,100,16,1
    const = const.expand(b, c, -1, -1)  # 1 100 16 2
    A = torch.cat([const, kp], dim=3)  # 1 100 16 3

    B[:, :, 0:1] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 1:2] = h * 0.5
    B[:, :, 2:3] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 3:4] = h * 0.5
    B[:, :, 4:5] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 5:6] = h * 0.5
    B[:, :, 6:7] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 7:8] = h * 0.5
    B[:, :, 8:9] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 9:10] = -h * 0.5
    B[:, :, 10:11] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 11:12] = -h * 0.5
    B[:, :, 12:13] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 13:14] = -h * 0.5
    B[:, :, 14:15] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 15:16] = -h * 0.5

    C[:, :, 0:1] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 1:2] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 2:3] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 3:4] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 4:5] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 5:6] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 6:7] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 7:8] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 8:9] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 9:10] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 10:11] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 11:12] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 12:13] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 13:14] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 14:15] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 15:16] = l * 0.5 * sinori + w * 0.5 * cosori

    B = B - kp_norm * C  # 1 100 16

    AT = A.permute(0, 1, 3, 2)  # 1 100 3 16
    AT = AT.view(b * c, 3, 16)  # 100 3 16
    A = A.view(b * c, 16, 3)  # 100 16 3
    B = B.view(b * c, 16, 1).float()  # 100 16 1
    # mask = mask.unsqueeze(2)

    pinv = torch.bmm(AT, A)  # 100 3 3
    pinv = torch.inverse(pinv)  # 100 3 3

    pinv = torch.bmm(pinv, AT)  # 100 3 16
    pinv = torch.bmm(pinv, B)  # 100 3 1
    pinv = pinv.view(b, c, 3, 1).squeeze(3)  # 1 100 3

    return pinv, rot_y, kps


def object_pose_decode(
        heat, wh, kps, dim, rot, prob=None, reg=None, hm_hp=None, hp_offset=None, K=100, meta=None, const=None):
    """The entry for Decoding object's pose with final bboxes, scores, key points, dimensions, heatmap score,
     yaw angle, position, probability, and classes.

        Args:
            heat (torch.Tensor): Predictions from model, batchsize * channels * h * w
            wh (torch.Tensor): widths and heights of bboxs
            kps (torch.Tensor): 1 * batchsize * 18
            dim (int): 1 * batchsize * 3 (h, w, l)
            rot (torch.Tensor): 1* batchsize * 8
            prob (torch.Tensor): probabilities
            reg (torch.Tensor): reg parameters
            hm_hp (torch.Tensor): heatmap hp parameters
            hp_offset (torch.Tensor): heatmap hp offsets
            K (int): top K value
            meta (dict): should contain keys below.
                - trans_output_inv: transition parameters for computing A in pinv's matrix.
                - calib: calibration for computing A in pinv's matrix.
            const (torch.Tensor): 1 * batchsize * 16 * 2

        Returns:
            torch.Tensor: 1 * batchsize * 28: detection, detection results containing final bboxes, scores, key points,
            dimensions, heatmap score, yaw angle, position, probability, and classes.
    """
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2

    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    # dim[:, :, 0] = torch.exp(dim[:, :, 0]) * 1.63
    # dim[:, :, 1] = torch.exp(dim[:, :, 1]) * 1.53
    # dim[:, :, 2] = torch.exp(dim[:, :, 2]) * 3.88
    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 8)
    prob = _transpose_and_gather_feat(prob, inds)[:, :, 0]
    prob = prob.view(batch, K, 1)
    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5
        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
        hm_score = hm_score.permute(0, 2, 1, 3).squeeze(3).contiguous()
    position, rot_y, kps_inv = gen_position(kps, dim, rot, meta, const)

    detections = torch.cat([bboxes, scores, kps_inv, dim, hm_score, rot_y, position, prob, clses], dim=2)

    return detections
