# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.structures import bbox3d2result, bbox3d_mapping_back, xywhr2xyxyr
from ..layers import nms_bev, nms_normal_bev


def merge_aug_bboxes_3d(aug_results, aug_batch_input_metas, test_cfg):
    """Merge augmented detection 3D bboxes and scores.

    Args:
        aug_results (list[dict]): The dict of detection results.
            The dict contains the following keys

            - bbox_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
        img_metas (list[dict]): Meta information of each sample.
        test_cfg (dict): Test config.

    Returns:
        dict: Bounding boxes results in cpu mode, containing merged results.

            - bbox_3d (:obj:`BaseInstance3DBoxes`): Merged detection bbox.
            - scores_3d (torch.Tensor): Merged detection scores.
            - labels_3d (torch.Tensor): Merged predicted box labels.
    """

    assert len(aug_results) == len(aug_batch_input_metas), \
        '"aug_results" should have the same length as "img_metas", got len(' \
        f'aug_results)={len(aug_results)} and ' \
        f'len(img_metas)={len(aug_batch_input_metas)}'

    recovered_bboxes = []
    recovered_scores = []
    recovered_labels = []

    for bboxes, input_info in zip(aug_results, aug_batch_input_metas):
        scale_factor = input_info['pcd_scale_factor']
        pcd_horizontal_flip = input_info['pcd_horizontal_flip']
        pcd_vertical_flip = input_info['pcd_vertical_flip']
        recovered_scores.append(bboxes['scores_3d'])
        recovered_labels.append(bboxes['labels_3d'])
        bboxes = bbox3d_mapping_back(bboxes['bbox_3d'], scale_factor,
                                     pcd_horizontal_flip, pcd_vertical_flip)
        recovered_bboxes.append(bboxes)

    aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes)
    aug_bboxes_for_nms = xywhr2xyxyr(aug_bboxes.bev)
    aug_scores = torch.cat(recovered_scores, dim=0)
    aug_labels = torch.cat(recovered_labels, dim=0)

    # TODO: use a more elegent way to deal with nms
    if test_cfg.get('use_rotate_nms', False):
        nms_func = nms_bev
    else:
        nms_func = nms_normal_bev

    merged_bboxes = []
    merged_scores = []
    merged_labels = []

    # Apply multi-class nms when merge bboxes
    if len(aug_labels) == 0:
        return bbox3d2result(aug_bboxes, aug_scores, aug_labels)

    for class_id in range(torch.max(aug_labels).item() + 1):
        class_inds = (aug_labels == class_id)
        bboxes_i = aug_bboxes[class_inds]
        bboxes_nms_i = aug_bboxes_for_nms[class_inds, :]
        scores_i = aug_scores[class_inds]
        labels_i = aug_labels[class_inds]
        if len(bboxes_nms_i) == 0:
            continue
        selected = nms_func(bboxes_nms_i, scores_i, test_cfg.nms_thr)

        merged_bboxes.append(bboxes_i[selected, :])
        merged_scores.append(scores_i[selected])
        merged_labels.append(labels_i[selected])

    merged_bboxes = merged_bboxes[0].cat(merged_bboxes)
    merged_scores = torch.cat(merged_scores, dim=0)
    merged_labels = torch.cat(merged_labels, dim=0)

    _, order = merged_scores.sort(0, descending=True)
    num = min(test_cfg.get('max_num', 500), len(aug_bboxes))
    order = order[:num]

    merged_bboxes = merged_bboxes[order]
    merged_scores = merged_scores[order]
    merged_labels = merged_labels[order]

    return bbox3d2result(merged_bboxes, merged_scores, merged_labels)
