import numpy as np


def fast_hist(preds, labels, num_classes):
    """Compute the confusion matrix for every batch.

    Args:
        preds (np.ndarray):  Prediction labels of points with shape of
        (num_points, ).
        labels (np.ndarray): Ground truth labels of points with shape of
        (num_points, ).
        num_classes (int): number of classes

    Returns:
        np.ndarray: Calculated confusion matrix.
    """

    k = (labels >= 0) & (labels < num_classes)
    bin_count = np.bincount(
        num_classes * labels[k].astype(int) + preds[k],
        minlength=num_classes**2)
    return bin_count[:num_classes**2].reshape(num_classes, num_classes)


def per_class_iou(hist):
    """Compute the per class iou.

    Args:
        hist(np.ndarray):  overall confusion martix
        (num_classes, num_classes ).
    Returns:
        np.ndarray: Calculated per class iou
    """

    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
