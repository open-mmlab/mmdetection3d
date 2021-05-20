import numpy as np
import torch
from torch.nn import functional as F


def _nms(heat, kernel=3):
    """Process heat with Non Maximum Suppression.
git
    Args:
        heat (torch.Tensor): Predictions from model in shape (batch_size *
            channels * height * width)
        kernel (int): kernel size for max pooling

    Returns:
        torch.Tensor (batch_size * channels * height * width): heatmap after
        NMS.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=40):
    """Select top k channels of scores, index, y's, and x's given scores.

    Args:
        scores (torch.Tensor): batch_size * channels * height * width
        K (int): top K value

    Returns:
        torch.Tensor (batch_size * channels * K): top k scores
        torch.Tensor (batch_size * channels * K): top k index
        torch.Tensor (batch_size * channels * K): top k y's
        torch.Tensor (batch_size * channels * K): top k x's
    """
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _gather_feat(feat, ind, mask=None):
    """Gather feat with assigned index, with index expanded to the third
    dimension.

    Args:
        feat (torch.Tensor): Predictions from model in shape (batch_size *
            channels * height * width)
        ind (torch.Tensor): index to be gathered

    Returns:
        torch.Tensor (batch_size * channels * height * width): gathered feat
    """

    # expand ind with the third dimension to be the same as feat
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """gather feat with assigned index, with feat be permuted with (0,2,3,1)
    and reshaped to three dimensions.

    Args:
        heat (torch.Tensor): Predictions from model in shape (batch_size *
            channels * height * width)
        ind (torch.Tensor): index to be gathered

    Returns:
        torch.Tensor (batch_size * channels * height * width): resulted feat
    """
    feat = feat.permute(0, 2, 3, 1)
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    """Select top k batches of scores, index, classes, y's, and x's given
    scores.

    Args:
        scores (torch.Tensor): batch_size * channels * height * width
        K (int): top K value

    Returns:
        torch.Tensor (batch_size * K): top k scores
        torch.Tensor (batch_size * K): top k index
        torch.Tensor (batch_size * K): top k classes
        torch.Tensor (batch_size * K): top k y's
        torch.Tensor (batch_size * K): top k x's
    """
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1),
                             topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def gen_position(kps, dim, rot, meta, const):
    """generate position from 2D frame to world frame.

    Args:
        kps (torch.Tensor): batch_size * boxes * 18
        dim (int): batch_size * boxes * 3 (height, width, length)
        rot (torch.Tensor): batch_size * boxes * 8
        meta (dict): should contain keys below.

            - trans_output_inv: transition parameters for computing
                matrix_A in pinv's matrix.
            - calib: calibration for computing matrix_A in pinv's matrix.
        const (torch.Tensor): batch_size * boxes * 16 * 2

    Returns:
        torch.Tensor (batch_size * boxes * 3): pinv, coordinates in world
            frame
        torch.Tensor (batch_size * boxes * 1): rot_y, yaw angle
        torch.Tensor (batch_size * boxes * 18): kps, key points coordinates
    """
    batch = kps.size(0)  # batch, number of pictures, usually 1.
    boxes = kps.size(1)  # boxes
    out_inv = meta['trans_output_inv']  # 1 2 3
    calib = meta['calib']  # 1 3 4

    out_inv = out_inv.unsqueeze(1)  # 1 1 2 3
    out_inv = out_inv.expand(batch, boxes, -1, -1).contiguous().view(-1, 2, 3)\
        .float()  # boxes 2 3
    kps = kps.view(batch, boxes, -1, 2).permute(0, 1, 3, 2)  # batch boxes 2 9
    # hom = torch.ones(batch, boxes, 1, 9).cuda()  # batch boxes 1 9
    hom = kps.new_ones(batch, boxes, 1, 9)  # batch boxes 1 9
    kps = torch.cat((kps, hom), dim=2).view(-1, 3, 9)  # boxes 3 9

    # batch boxes 2 9     [[xxxxxxx...],[yyyyyy...]]
    kps = torch.bmm(out_inv, kps).view(batch, boxes, 2, 9)
    # batch boxes 18 [xyxyxyxyx....]
    kps = kps.permute(0, 1, 3, 2).view(batch, boxes, -1)
    # calib2 batch boxes 1
    calib2 = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1]
    alpha_idx = rot[:, :, 1] > rot[:, :, 5]  # batch boxes
    alpha_idx = alpha_idx.float()  # batch boxes
    # alpha1 Shape: [batch boxes]
    alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)
    alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)  # batch boxes
    alpna_pre = alpna_pre.unsqueeze(2)  # batch boxes

    # batch boxes 1
    rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3],
                                    calib2)
    rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi  # batch boxes 1
    rot_y[rot_y < -np.pi] = rot_y[rot_y < -np.pi] + 2 * np.pi

    calib = calib.unsqueeze(1)  # 1 1 3 4
    calib = calib.expand(batch, boxes, -1, -1).contiguous()  # batch boxes 3 4
    kpoint = kps[:, :, :16]  # batch boxes 16
    f = calib[:, :, 0, 0].unsqueeze(2)  # batch boxes 1
    f = f.expand_as(kpoint)  # batch boxes 16
    # batch boxes 1 , batch boxes 1
    cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)
    cxy = torch.cat((cx, cy), dim=2)  # 1 boxes 2
    cxy = cxy.repeat(1, 1, 8)  # 1 boxes 16
    kp_norm = (kpoint - cxy) / f  # 1 boxes 16

    height = dim[:, :, 0:1]  # batch boxes 1
    width = dim[:, :, 1:2]  # batch boxes 1
    length = dim[:, :, 2:3]  # batch boxes 1
    cosori = torch.cos(rot_y)  # batch boxes 1
    sinori = torch.sin(rot_y)  # batch boxes 1

    matrix_B = torch.zeros_like(kpoint)  # batch boxes 16
    matrix_C = torch.zeros_like(kpoint)  # batch boxes 16

    kp = kp_norm.unsqueeze(3)  # batch,boxes,16,1
    const = const.expand(batch, boxes, -1, -1)  # batch boxes 16 2
    matrix_A = torch.cat([const, kp], dim=3)  # batch boxes 16 3

    matrix_B[:, :, 0:1] = length * 0.5 * cosori + width * 0.5 * sinori
    matrix_B[:, :, 1:2] = height * 0.5
    matrix_B[:, :, 2:3] = length * 0.5 * cosori - width * 0.5 * sinori
    matrix_B[:, :, 3:4] = height * 0.5
    matrix_B[:, :, 4:5] = -length * 0.5 * cosori - width * 0.5 * sinori
    matrix_B[:, :, 5:6] = height * 0.5
    matrix_B[:, :, 6:7] = -length * 0.5 * cosori + width * 0.5 * sinori
    matrix_B[:, :, 7:8] = height * 0.5
    matrix_B[:, :, 8:9] = length * 0.5 * cosori + width * 0.5 * sinori
    matrix_B[:, :, 9:10] = -height * 0.5
    matrix_B[:, :, 10:11] = length * 0.5 * cosori - width * 0.5 * sinori
    matrix_B[:, :, 11:12] = -height * 0.5
    matrix_B[:, :, 12:13] = -length * 0.5 * cosori - width * 0.5 * sinori
    matrix_B[:, :, 13:14] = -height * 0.5
    matrix_B[:, :, 14:15] = -length * 0.5 * cosori + width * 0.5 * sinori
    matrix_B[:, :, 15:16] = -height * 0.5

    matrix_C[:, :, 0:1] = -length * 0.5 * sinori + width * 0.5 * cosori
    matrix_C[:, :, 1:2] = -length * 0.5 * sinori + width * 0.5 * cosori
    matrix_C[:, :, 2:3] = -length * 0.5 * sinori - width * 0.5 * cosori
    matrix_C[:, :, 3:4] = -length * 0.5 * sinori - width * 0.5 * cosori
    matrix_C[:, :, 4:5] = length * 0.5 * sinori - width * 0.5 * cosori
    matrix_C[:, :, 5:6] = length * 0.5 * sinori - width * 0.5 * cosori
    matrix_C[:, :, 6:7] = length * 0.5 * sinori + width * 0.5 * cosori
    matrix_C[:, :, 7:8] = length * 0.5 * sinori + width * 0.5 * cosori
    matrix_C[:, :, 8:9] = -length * 0.5 * sinori + width * 0.5 * cosori
    matrix_C[:, :, 9:10] = -length * 0.5 * sinori + width * 0.5 * cosori
    matrix_C[:, :, 10:11] = -length * 0.5 * sinori - width * 0.5 * cosori
    matrix_C[:, :, 11:12] = -length * 0.5 * sinori - width * 0.5 * cosori
    matrix_C[:, :, 12:13] = length * 0.5 * sinori - width * 0.5 * cosori
    matrix_C[:, :, 13:14] = length * 0.5 * sinori - width * 0.5 * cosori
    matrix_C[:, :, 14:15] = length * 0.5 * sinori + width * 0.5 * cosori
    matrix_C[:, :, 15:16] = length * 0.5 * sinori + width * 0.5 * cosori

    matrix_B = matrix_B - kp_norm * matrix_C  # 1 boxes 16

    AT = matrix_A.permute(0, 1, 3, 2)  # 1 boxes 3 16
    AT = AT.view(batch * boxes, 3, 16)  # boxes 3 16
    matrix_A = matrix_A.view(batch * boxes, 16, 3)  # boxes 16 3
    matrix_B = matrix_B.view(batch * boxes, 16, 1).float()  # boxes 16 1
    # mask = mask.unsqueeze(2)

    pinv = torch.bmm(AT, matrix_A)  # boxes 3 3
    pinv = torch.inverse(pinv)  # boxes 3 3

    pinv = torch.bmm(pinv, AT)  # boxes 3 16
    pinv = torch.bmm(pinv, matrix_B)  # boxes 3 1
    pinv = pinv.view(batch, boxes, 3, 1).squeeze(3)  # 1 boxes 3

    return pinv, rot_y, kps


def object_pose_decode(heat,
                       wh,
                       kps,
                       dim,
                       rot,
                       prob=None,
                       reg=None,
                       hm_hp=None,
                       hp_offset=None,
                       K=100,
                       meta=None,
                       const=None):
    """The entry for Decoding object's pose with final bboxes, scores, key
    points, dimensions, heatmap score, yaw angle, position, probability, and
    classes.

    Args:
        heat (torch.Tensor): Predictions from model in shape (batch_size *
            channels * height * width)
        wh (torch.Tensor): widths and heights of bboxs
        kps (torch.Tensor): 1 * batch_size * 18
        dim (int): 1 * batch_size * 3 (height, width, length)
        rot (torch.Tensor): 1* batch_size * 8
        prob (torch.Tensor): probabilities
        reg (torch.Tensor): reg parameters
        hm_hp (torch.Tensor): heatmap hp parameters
        hp_offset (torch.Tensor): heatmap hp offsets
        K (int): top K value
        meta (dict): should contain keys below.

            - trans_output_inv: transition parameters for computing
                matrix_A in pinv's matrix.
            - calib: calibration for computing matrix_A in pinv's matrix.
        const (torch.Tensor): 1 * batch_size * 16 * 2

    Returns:
        torch.Tensor (1 * batch_size * 28): detection, detection results
            containing final bboxes, scores, key points, dimensions,
            heatmap score, yaw angle, position, probability, and classes.
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

    bboxes = torch.cat([
        xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2,
        ys + wh[..., 1:2] / 2
    ],
                       dim=2)
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
        kps = kps.view(batch, K, num_joints, 2).permute(0, 2, 1,
                                                        3)  # batch x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        # batch x J x K
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)
        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(hp_offset,
                                                   hm_inds.view(batch, -1))
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
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(2).expand(
            batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps)**2).sum(dim=4)**0.5)
        min_dist, min_ind = dist.min(dim=3)  # batch x J x K
        # batch x J x K x 1
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1,
                               1).expand(batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        length = bboxes[:, :, 0].view(batch, 1, K, 1)\
            .expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1)\
            .expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1)\
            .expand(batch, num_joints, K, 1)
        batch = bboxes[:, :, 3].view(batch, 1, K, 1)\
            .expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < length) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > batch) + \
               (hm_score < thresh) + \
               (min_dist > (torch.max(batch - t, r - length) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).view(batch, K, num_joints * 2)
        hm_score = hm_score.permute(0, 2, 1, 3).squeeze(3)
    position, rot_y, kps_inv = gen_position(kps, dim, rot, meta, const)

    detections = torch.cat(
        [bboxes, scores, kps_inv, dim, hm_score, rot_y, position, prob, clses],
        dim=2)

    return detections
