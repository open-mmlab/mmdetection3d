import torch


def bbox3d_mapping_back(bboxes, scale_factor, flip_horizontal, flip_vertical):
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = bboxes.clone()
    if flip_horizontal:
        new_bboxes.flip('horizontal')
    if flip_vertical:
        new_bboxes.flip('vertical')
    new_bboxes.scale(1 / scale_factor)

    return new_bboxes


def bbox3d2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[torch.Tensor]): a list of bboxes
            corresponding to a batch of images.

    Returns:
        torch.Tensor: shape (n, c), [batch_ind, x, y ...].
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes], dim=-1)
        else:
            rois = torch.zeros_like(bboxes)
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def bbox3d2result(bboxes, scores, labels):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): shape (n, 5)
        labels (torch.Tensor): shape (n, )
        scores (torch.Tensor): shape (n, )

    Returns:
        dict(torch.Tensor): Bbox results in cpu mode.
    """
    return dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())
