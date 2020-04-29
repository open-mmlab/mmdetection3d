import mmcv

from mmdet3d.datasets.pipelines.indoor_loading import IndoorLoadData


def test_indoor_load_data():
    train_info = mmcv.load('./tests/data/sunrgbd/sunrgbd_infos_train.pkl')
    sunrgbd_load_data = IndoorLoadData('sunrgbd', False, True, [0.5, 0.5, 0.5])
    sunrgbd_results = dict()
    sunrgbd_results['data_path'] = './tests/data/sunrgbd/sunrgbd_trainval'
    sunrgbd_results['info'] = train_info[0]
    sunrgbd_results = sunrgbd_load_data(sunrgbd_results)
    point_cloud = sunrgbd_results.get('point_cloud', None)
    gt_boxes = sunrgbd_results.get('gt_boxes', None)
    gt_classes = sunrgbd_results.get('gt_classes', None)
    gt_boxes_mask = sunrgbd_results.get('gt_boxes_mask', None)
    assert point_cloud.shape == (50000, 4)
    assert gt_boxes.shape == (3, 7)
    assert gt_classes.shape == (3, 1)
    assert gt_boxes_mask.shape == (3, 1)
