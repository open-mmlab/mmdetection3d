import mmcv

from mmdet3d.datasets.pipelines.indoor_loading import IndoorLoadData


def test_indoor_load_data():
    sunrgbd_train_info = mmcv.load(
        './tests/data/sunrgbd/sunrgbd_infos_train.pkl')
    sunrgbd_load_train_data = IndoorLoadData('sunrgbd', False, True,
                                             [0.5, 0.5, 0.5])
    sunrgbd_train_results = dict()
    sunrgbd_train_results[
        'data_path'] = './tests/data/sunrgbd/sunrgbd_trainval'
    sunrgbd_train_results['info'] = sunrgbd_train_info[0]
    sunrgbd_train_results = sunrgbd_load_train_data(sunrgbd_train_results)
    sunrgbd_train_point_cloud = sunrgbd_train_results.get('point_cloud', None)
    sunrgbd_train_gt_boxes = sunrgbd_train_results.get('gt_boxes', None)
    sunrgbd_train_gt_classes = sunrgbd_train_results.get('gt_classes', None)
    sunrgbd_train_gt_boxes_mask = sunrgbd_train_results.get(
        'gt_boxes_mask', None)
    assert sunrgbd_train_point_cloud.shape == (50000, 4)
    assert sunrgbd_train_gt_boxes.shape == (3, 7)
    assert sunrgbd_train_gt_classes.shape == (3, 1)
    assert sunrgbd_train_gt_boxes_mask.shape == (3, 1)

    scannet_val_info = mmcv.load('./tests/data/sunrgbd/sunrgbd_infos_val.pkl')
    scannet_load_val_data = IndoorLoadData('sunrgbd', False, True,
                                           [0.5, 0.5, 0.5])
    scannet_val_results = dict()
    scannet_val_results['data_path'] = './tests/data/sunrgbd/sunrgbd_trainval'
    scannet_val_results['info'] = scannet_val_info[0]
    scannet_val_results = scannet_load_val_data(scannet_val_results)
    scannet_val_point_cloud = scannet_val_results.get('point_cloud', None)
    scannet_val_gt_boxes = scannet_val_results.get('gt_boxes', None)
    scannet_val_gt_classes = scannet_val_results.get('gt_classes', None)
    scannet_val_gt_boxes_mask = scannet_val_results.get('gt_boxes_mask', None)
    assert scannet_val_point_cloud.shape == (50000, 4)
    assert scannet_val_gt_boxes.shape == (3, 7)
    assert scannet_val_gt_classes.shape == (3, 1)
    assert scannet_val_gt_boxes_mask.shape == (3, 1)

    sunrgbd_train_info = mmcv.load(
        './tests/data/sunrgbd/sunrgbd_infos_train.pkl')
    sunrgbd_load_train_data = IndoorLoadData('sunrgbd', False, True,
                                             [0.5, 0.5, 0.5])
    sunrgbd_train_results = dict()
    sunrgbd_train_results[
        'data_path'] = './tests/data/sunrgbd/sunrgbd_trainval'
    sunrgbd_train_results['info'] = sunrgbd_train_info[0]
    sunrgbd_train_results = sunrgbd_load_train_data(sunrgbd_train_results)
    sunrgbd_train_point_cloud = sunrgbd_train_results.get('point_cloud', None)
    sunrgbd_train_gt_boxes = sunrgbd_train_results.get('gt_boxes', None)
    sunrgbd_train_gt_classes = sunrgbd_train_results.get('gt_classes', None)
    sunrgbd_train_gt_boxes_mask = sunrgbd_train_results.get(
        'gt_boxes_mask', None)
    assert sunrgbd_train_point_cloud.shape == (50000, 4)
    assert sunrgbd_train_gt_boxes.shape == (3, 7)
    assert sunrgbd_train_gt_classes.shape == (3, 1)
    assert sunrgbd_train_gt_boxes_mask.shape == (3, 1)

    scannet_val_info = mmcv.load(
        './tests/data/scannet/scannet_infos_train.pkl')
    scannet_load_val_data = IndoorLoadData('scannet', False, True,
                                           [0.5, 0.5, 0.5])
    scannet_val_results = dict()
    scannet_val_results[
        'data_path'] = './tests/data/scannet/scannet_train_instance_data'
    scannet_val_results['info'] = scannet_val_info[0]
    scannet_val_results = scannet_load_val_data(scannet_val_results)
    scannet_val_point_cloud = scannet_val_results.get('point_cloud', None)
    scannet_val_gt_boxes = scannet_val_results.get('gt_boxes', None)
    scannet_val_gt_classes = scannet_val_results.get('gt_classes', None)
    scannet_val_gt_boxes_mask = scannet_val_results.get('gt_boxes_mask', None)
    scannet_pcl_color = scannet_val_results.get('pcl_color', None)
    scannet_instance_labels = scannet_val_results.get('instance_labels', None)
    scannet_semantic_labels = scannet_val_results.get('semantic_labels', None)
    assert scannet_val_point_cloud.shape == (50000, 4)
    assert scannet_val_gt_boxes.shape == (27, 6)
    assert scannet_val_gt_classes.shape == (27, 1)
    assert scannet_val_gt_boxes_mask.shape == (27, 1)
    assert scannet_pcl_color.shape == (50000, 3)
    assert scannet_instance_labels.shape == (50000, )
    assert scannet_semantic_labels.shape == (50000, )

    scannet_val_info = mmcv.load('./tests/data/scannet/scannet_infos_val.pkl')
    scannet_load_val_data = IndoorLoadData('scannet', False, True,
                                           [0.5, 0.5, 0.5])
    scannet_val_results = dict()
    scannet_val_results[
        'data_path'] = './tests/data/scannet/scannet_train_instance_data'
    scannet_val_results['info'] = scannet_val_info[0]
    scannet_val_results = scannet_load_val_data(scannet_val_results)
    scannet_val_point_cloud = scannet_val_results.get('point_cloud', None)
    scannet_val_gt_boxes = scannet_val_results.get('gt_boxes', None)
    scannet_val_gt_classes = scannet_val_results.get('gt_classes', None)
    scannet_val_gt_boxes_mask = scannet_val_results.get('gt_boxes_mask', None)
    assert scannet_val_point_cloud.shape == (50000, 4)
    assert scannet_val_gt_boxes.shape == (28, 6)
    assert scannet_val_gt_classes.shape == (28, 1)
    assert scannet_val_gt_boxes_mask.shape == (28, 1)
