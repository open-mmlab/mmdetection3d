import numpy as np

from mmdet3d.core.evaluation.kitti_utils.eval import eval_class


def test_eval_class():
    gt_name = np.array(
        ['Pedestrian', 'Cyclist', 'Car', 'Car', 'Car', 'DontCare', 'DontCare'])
    gt_truncated = np.array([0., 0., 0., -1., -1., -1., -1.])
    gt_occluded = np.array([0, 0, 3, -1, -1, -1, -1])
    gt_alpha = np.array([-1.57, 1.85, -1.65, -10., -10., -10., -10.])
    gt_bbox = np.array([[674.9179, 165.48549, 693.23694, 193.42134],
                        [676.21954, 165.70988, 691.63745, 193.83748],
                        [389.4093, 182.48041, 421.49072, 202.13422],
                        [232.0577, 186.16724, 301.94623, 217.4024],
                        [758.6537, 172.98509, 816.32434, 212.76743],
                        [532.37, 176.35, 542.68, 185.27],
                        [559.62, 175.83, 575.4, 183.15]])
    gt_anno = dict(
        name=gt_name,
        truncated=gt_truncated,
        occluded=gt_occluded,
        alpha=gt_alpha,
        bbox=gt_bbox)

    dt_name = np.array(['Pedestrian', 'Cyclist', 'Car', 'Car', 'Car'])
    dt_truncated = np.array([0., 0., 0., 0., 0.])
    dt_occluded = np.array([0, 0, 0, 0, 0])
    dt_alpha = np.array([1.0744612, 1.2775835, 1.82563, 2.1145396, -1.7676563])
    dt_bbox = np.array([[674.9179, 165.48549, 693.23694, 193.42134],
                        [676.21954, 165.70988, 691.63745, 193.83748],
                        [389.4093, 182.48041, 421.49072, 202.13422],
                        [232.0577, 186.16724, 301.94623, 217.4024],
                        [758.6537, 172.98509, 816.32434, 212.76743]])
    dt_score = np.array(
        [0.18151495, 0.57920843, 0.27795696, 0.23100418, 0.21541929])
    dt_anno = dict(
        name=dt_name,
        truncated=dt_truncated,
        occluded=dt_occluded,
        alpha=dt_alpha,
        bbox=dt_bbox,
        score=dt_score)
    current_classes = [1, 2, 0]
    difficultys = [0, 1, 2]
    metric = 0
    min_overlaps = np.array([[[0.5, 0.5, 0.7], [0.5, 0.5, 0.7],
                              [0.5, 0.5, 0.7]],
                             [[0.5, 0.5, 0.7], [0.25, 0.25, 0.5],
                              [0.25, 0.25, 0.5]]])

    ret_dict = eval_class([gt_anno], [dt_anno], current_classes, difficultys,
                          metric, min_overlaps, True, 1)
    recall_sum = np.sum(ret_dict['recall'])
    precision_sum = np.sum(ret_dict['precision'])
    orientation_sum = np.sum(ret_dict['orientation'])
    assert abs(recall_sum - 16) < 1e-5
    assert abs(precision_sum - 16) < 1e-5
    assert abs(orientation_sum - 10.252829201850309) < 1e-5
