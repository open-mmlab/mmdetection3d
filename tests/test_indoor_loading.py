import mmcv

from mmdet3d.datasets.pipelines.indoor_loading import IndoorLoadData


def test_indoor_load_data():
    sunrgbd_info = mmcv.load('./tests/data/sunrgbd/sunrgbd_infos.pkl')
    sunrgbd_load_data = IndoorLoadData('sunrgbd', False, True, [0.5, 0.5, 0.5])
    sunrgbd_results = dict()
    sunrgbd_results['data_path'] = './tests/data/sunrgbd/sunrgbd_trainval'
    sunrgbd_results['info'] = sunrgbd_info[0]
    sunrgbd_results = sunrgbd_load_data(sunrgbd_results)
    sunrgbd_point_cloud = sunrgbd_results.get('point_cloud', None)
    sunrgbd_gt_boxes = sunrgbd_results.get('gt_boxes', None)
    sunrgbd_gt_classes = sunrgbd_results.get('gt_classes', None)
    sunrgbd_gt_boxes_mask = sunrgbd_results.get('gt_boxes_mask', None)
    assert sunrgbd_point_cloud.shape == (1000, 4)
    assert sunrgbd_gt_boxes.shape == (3, 7)
    assert sunrgbd_gt_classes.shape == (3, 1)
    assert sunrgbd_gt_boxes_mask.shape == (3, 1)

    scannet_info = mmcv.load('./tests/data/scannet/scannet_infos.pkl')
    scannet_load_data = IndoorLoadData('scannet', False, True, [0.5, 0.5, 0.5])
    scannet_results = dict()
    scannet_results[
        'data_path'] = './tests/data/scannet/scannet_train_instance_data'
    scannet_results['info'] = scannet_info[0]
    scannet_results = scannet_load_data(scannet_results)
    scannet_point_cloud = scannet_results.get('point_cloud', None)
    scannet_gt_boxes = scannet_results.get('gt_boxes', None)
    scannet_gt_classes = scannet_results.get('gt_classes', None)
    scannet_gt_boxes_mask = scannet_results.get('gt_boxes_mask', None)
    scannet_pcl_color = scannet_results.get('pcl_color', None)
    scannet_instance_labels = scannet_results.get('instance_labels', None)
    scannet_semantic_labels = scannet_results.get('semantic_labels', None)
    assert scannet_point_cloud.shape == (1000, 4)
    assert scannet_gt_boxes.shape == (27, 6)
    assert scannet_gt_classes.shape == (27, 1)
    assert scannet_gt_boxes_mask.shape == (27, 1)
    assert scannet_pcl_color.shape == (1000, 3)
    assert scannet_instance_labels.shape == (1000, )
    assert scannet_semantic_labels.shape == (1000, )
