# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import pickle

import torch
# --- IoU BEV (CUDA op, still in mmcv.ops) ---
from mmcv.ops import boxes_iou_bev
# --- mkdir_or_exist (old: mmcv → new: mmengine.utils) ---
from mmengine.utils import mkdir_or_exist

# --- NMS functions ---
from mmdet3d.models.layers import nms_bev, nms_normal_bev
# --- Structures (old: mmdet3d.core.bbox) ---
from mmdet3d.structures import (LiDARInstance3DBoxes, bbox3d2result,
                                bbox3d_mapping_back, xywhr2xyxyr)

ensemble = False


def merge_aug_bboxes_3d(aug_results, img_metas, test_cfg):
    """Merge augmented detection 3D bboxes and scores.

    Args:
        aug_results (list[dict]): The dict of detection results.
            The dict contains the following keys

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
        img_metas (list[dict]): Meta information of each sample.
        test_cfg (dict): Test config.

    Returns:
        dict: Bounding boxes results in cpu mode, containing merged results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Merged detection bbox.
            - scores_3d (torch.Tensor): Merged detection scores.
            - labels_3d (torch.Tensor): Merged predicted box labels.
    """

    if ensemble:
        import glob
        ensemble_folder = './merge_augs/*'
        aug_bboxes = []
        aug_bboxes_for_nms = []
        aug_scores = []
        aug_labels = []

        ensemble_models = glob.glob(ensemble_folder)

        for ensemble_model in ensemble_models:
            pkl_path = (f'{ensemble_model}/'
                        f'sampleidx_{img_metas[0][0]["sample_idx"]}.pkl')
            with open(pkl_path, 'rb') as f:
                temp = pickle.load(f)

            aug_bboxes.append(
                torch.as_tensor(
                    temp['aug_bboxes'], dtype=torch.float32, device='cuda'))
            aug_bboxes_for_nms.append(
                torch.as_tensor(
                    temp['aug_bboxes_for_nms'],
                    dtype=torch.float32,
                    device='cuda'))
            aug_scores.append(
                torch.as_tensor(
                    temp['aug_scores'], dtype=torch.float32, device='cuda'))
            aug_labels.append(
                torch.as_tensor(
                    temp['aug_labels'], dtype=torch.int32, device='cuda'))

        aug_bboxes = torch.cat(aug_bboxes, dim=0)
        aug_bboxes_for_nms = torch.cat(aug_bboxes_for_nms, dim=0)
        aug_scores = torch.cat(aug_scores, dim=0)
        aug_labels = torch.cat(aug_labels, dim=0)

        aug_bboxes = LiDARInstance3DBoxes(
            aug_bboxes, box_dim=aug_bboxes.shape[-1])
    else:
        if 'temp_result_folder' in test_cfg:
            temp_folder = './merge_augs/' + test_cfg.temp_result_folder
        else:
            temp_folder = './merge_augs_initial_results/'

        mkdir_or_exist(temp_folder)

        if aug_results is None:
            pkl_path = (f'{temp_folder}/'
                        f'sampleidx_{img_metas[0][0]["sample_idx"]}.pkl')
            with open(pkl_path, 'rb') as f:
                temp = pickle.load(f)

            aug_bboxes = torch.as_tensor(
                temp['aug_bboxes'], dtype=torch.float32, device='cuda')
            aug_bboxes_for_nms = torch.as_tensor(
                temp['aug_bboxes_for_nms'], dtype=torch.float32, device='cuda')
            aug_scores = torch.as_tensor(
                temp['aug_scores'], dtype=torch.float32, device='cuda')
            aug_labels = torch.as_tensor(
                temp['aug_labels'], dtype=torch.int32, device='cuda')

            aug_bboxes = LiDARInstance3DBoxes(
                aug_bboxes, box_dim=aug_bboxes.shape[-1])
        else:
            assert len(aug_results) == len(img_metas), \
                '"aug_results" should have the same length as "img_metas", ' \
                f'got len(aug_results)={len(aug_results)} and ' \
                f'len(img_metas)={len(img_metas)}'

            recovered_bboxes = []
            recovered_scores = []
            recovered_labels = []

            for aug_idx, (bboxes,
                          img_info) in enumerate(zip(aug_results, img_metas)):
                scale_factor = img_info[0]['pcd_scale_factor']
                pcd_horizontal_flip = img_info[0]['pcd_horizontal_flip']
                pcd_vertical_flip = img_info[0]['pcd_vertical_flip']
                recovered_scores.append(bboxes['scores_3d'])
                recovered_labels.append(bboxes['labels_3d'])
                bboxes = bbox3d_mapping_back(bboxes['boxes_3d'], scale_factor,
                                             pcd_horizontal_flip,
                                             pcd_vertical_flip)
                recovered_bboxes.append(bboxes)

            aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes)
            aug_bboxes_for_nms = xywhr2xyxyr(aug_bboxes.bev)
            aug_scores = torch.cat(recovered_scores, dim=0)
            aug_labels = torch.cat(recovered_labels, dim=0)

            if True:
                temp = dict()
                temp['aug_bboxes'] = aug_bboxes.tensor.cpu().numpy()
                temp['aug_bboxes_for_nms'] = aug_bboxes_for_nms.cpu().numpy()
                temp['aug_scores'] = aug_scores.cpu().numpy()
                temp['aug_labels'] = aug_labels.cpu().numpy()
                pkl_path = (f'{temp_folder}/'
                            f'sampleidx_{img_metas[0][0]["sample_idx"]}.pkl')
                with open(pkl_path, 'wb') as f:
                    pickle.dump(temp, f)

    test_cfg = test_cfg.copy()
    # Extra NMS settings
    test_cfg['nms_type'] = 'rotate'
    test_cfg['use_rotate_nms'] = True
    test_cfg['max_num'] = 500
    test_cfg['nms_thr'] = 0.1
    test_cfg['score_threshold'] = 0.05

    # Select NMS function
    # OLD: nms_gpu / nms_normal_gpu
    # NEW: nms_bev / nms_normal_bev
    if test_cfg.use_rotate_nms:
        nms_func = nms_bev
    else:
        nms_func = nms_normal_bev

    merged_bboxes = []
    merged_scores = []
    merged_labels = []

    # Apply multi-class NMS when merging bboxes
    if len(aug_labels) == 0:
        return bbox3d2result(aug_bboxes, aug_scores, aug_labels)

    num_classes = torch.max(aug_labels).item() + 1

    for class_id in range(num_classes):
        class_inds = (aug_labels == class_id)
        bboxes_i = aug_bboxes[class_inds]
        bboxes_nms_i = aug_bboxes_for_nms[class_inds, :]
        scores_i = aug_scores[class_inds]
        labels_i = aug_labels[class_inds]

        if len(bboxes_nms_i) == 0:
            continue

        selected = nms_func(bboxes_nms_i, scores_i, test_cfg.nms_thr)

        if True:  # voting
            vote_iou_thresh = 0.65
            use_voting_scores = False

            selected_bboxes = bboxes_i[selected, :]
            selected_scores = scores_i[selected]
            selected_labels = labels_i[selected]

            iou = boxes_iou_bev(xywhr2xyxyr(selected_bboxes.bev), bboxes_nms_i)

            iou[iou < vote_iou_thresh] = 0.

            voted_bboxes = (
                (iou[:, :, None] * bboxes_i.tensor[None]).sum(dim=1) /
                (iou[:, :, None].sum(dim=1) + 1e-6))
            voted_bboxes[:, 6] = torch.atan2(
                (iou * torch.sin(bboxes_i.tensor[None, :, 6])).sum(dim=1) /
                (iou.sum(dim=1) + 1e-6),
                (iou * torch.cos(bboxes_i.tensor[None, :, 6])).sum(dim=1) /
                (iou.sum(dim=1) + 1e-6))

            voted_bboxes = LiDARInstance3DBoxes(
                voted_bboxes, box_dim=voted_bboxes.shape[-1])

            selected_bboxes = voted_bboxes
            if use_voting_scores:
                voted_scores = ((iou * scores_i[None]).sum(dim=1) /
                                iou.sum(dim=1))
                selected_scores = voted_scores

        merged_bboxes.append(selected_bboxes)
        merged_scores.append(selected_scores)
        merged_labels.append(selected_labels)

    if len(merged_bboxes) == 0:
        return bbox3d2result(aug_bboxes[:0], aug_scores[:0], aug_labels[:0])

    merged_bboxes = merged_bboxes[0].cat(merged_bboxes)
    merged_scores = torch.cat(merged_scores, dim=0)
    merged_labels = torch.cat(merged_labels, dim=0)

    _, order = merged_scores.sort(0, descending=True)
    num = min(test_cfg.max_num, len(aug_bboxes))
    order = order[:num]

    merged_bboxes = merged_bboxes[order]
    merged_scores = merged_scores[order]
    merged_labels = merged_labels[order]

    return bbox3d2result(merged_bboxes, merged_scores, merged_labels)
