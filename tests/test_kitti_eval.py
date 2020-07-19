import numpy as np

from mmdet3d.core.evaluation.kitti_utils.eval import clean_data


def test_clean_data():
    gt_name = np.array(
        ['Truck'
         'Car'
         'Cyclist'
         'DontCare'
         'DontCare'
         'DontCare'
         'DontCare'])
    gt_truncated = np.array([0., 0., 0., -1., -1., -1., -1.])
    gt_occluded = np.array([0, 0, 3, -1, -1, -1, -1])
    gt_bbox = np.array([[599.41, 156.4, 629.75, 189.25],
                        [387.63, 181.54, 423.81, 203.12],
                        [676.6, 163.95, 688.98, 193.93],
                        [503.89, 169.71, 590.61, 190.13],
                        [511.35, 174.96, 527.81, 187.45],
                        [532.37, 176.35, 542.68, 185.27],
                        [559.62, 175.83, 575.4, 183.15]])
    gt_annos = dict(
        name=gt_name,
        truncated=gt_truncated,
        occluded=gt_occluded,
        bbox=gt_bbox)
    clean_data(gt_annos, gt_annos, 1, 0)
    dt_name = np.array(['Pedestrian' 'Cyclist' 'Car' 'Car' 'Car'])
    dt_truncated = np.array([0., 0., 0., 0., 0.])
    dt_occluded = np.array([0, 0, 0, 0, 0])
    dt_bbox = np.array([[674.9179, 165.48549, 693.23694, 193.42134],
                        [676.21954, 165.70988, 691.63745, 193.83748],
                        [389.4093, 182.48041, 421.49072, 202.13422],
                        [232.0577, 186.16724, 301.94623, 217.4024],
                        [758.6537, 172.98509, 816.32434, 212.76743]])
    dt_annos = dict(
        name=dt_name,
        truncated=dt_truncated,
        occluded=dt_occluded,
        bbox=dt_bbox)
    num_valid_gt, ignored_gt, ignored_dt, dc_bboxes = clean_data(
        gt_annos, dt_annos, 1, 0)
    assert num_valid_gt == 0
    assert ignored_gt == [-1]
    assert ignored_dt == [1]
    assert dc_bboxes == []
