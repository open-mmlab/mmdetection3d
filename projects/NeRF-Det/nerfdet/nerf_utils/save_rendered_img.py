# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity


def compute_psnr_from_mse(mse):
    return -10.0 * torch.log(mse) / np.log(10.0)


def compute_psnr(pred, target, mask=None):
    """Compute psnr value (we assume the maximum pixel value is 1)."""
    if mask is not None:
        pred, target = pred[mask], target[mask]
    mse = ((pred - target)**2).mean()
    return compute_psnr_from_mse(mse).cpu().numpy()


def compute_ssim(pred, target, mask=None):
    """Computes Masked SSIM following the neuralbody paper."""
    assert pred.shape == target.shape and pred.shape[-1] == 3
    if mask is not None:
        x, y, w, h = cv2.boundingRect(mask.cpu().numpy().astype(np.uint8))
        pred = pred[y:y + h, x:x + w]
        target = target[y:y + h, x:x + w]
    try:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), channel_axis=-1)
    except ValueError:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), multichannel=True)
    return ssim


def save_rendered_img(img_meta, rendered_results):
    filename = img_meta[0]['filename']
    scenes = filename.split('/')[-2]

    for ret in rendered_results:
        depth = ret['outputs_coarse']['depth']
        rgb = ret['outputs_coarse']['rgb']
        gt = ret['gt_rgb']
        gt_depth = ret['gt_depth']

    # save images
    psnr_total = 0
    ssim_total = 0
    rsme = 0
    for v in range(gt.shape[0]):
        rsme += ((depth[v] - gt_depth[v])**2).cpu().numpy()
        depth_ = ((depth[v] - depth[v].min()) /
                  (depth[v].max() - depth[v].min() + 1e-8)).repeat(1, 1, 3)
        img_to_save = torch.cat([rgb[v], gt[v], depth_], dim=1)
        image_path = os.path.join('nerf_vs_rebuttal', scenes)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        save_dir = os.path.join(image_path, 'view_' + str(v) + '.png')

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        image = np.uint8(img_to_save.cpu().numpy() * 255.0)
        psnr = compute_psnr(rgb[v], gt[v], mask=None)
        psnr_total += psnr
        ssim = compute_ssim(rgb[v], gt[v], mask=None)
        ssim_total += ssim
        image = cv2.putText(
            image, 'PSNR: ' + '%.2f' % compute_psnr(rgb[v], gt[v], mask=None),
            org, font, fontScale, color, thickness, cv2.LINE_AA)

        cv2.imwrite(save_dir, image)

    return psnr_total / gt.shape[0], ssim_total / gt.shape[0], rsme / gt.shape[
        0]
