import numpy as np

from mmdet3d.datasets.pipelines.indoor_sample import PointSample


def test_indoor_sample():
    scannet_sample_points = PointSample('scannet', 1)
    scannet_results = dict()
    scannet_results['points'] = np.array(
        [[1.6110241e+00, -1.6903955e-01, 5.8115810e-01, 5.9897250e-01],
         [1.3978075e+00, 4.2035791e-01, 3.8729519e-01, 4.0510958e-01]])
    scannet_results['instance_labels'] = np.array([9, 3])
    scannet_results['pcl_color'] = np.array([[29.0, 142.0, 122.0],
                                             [21., 17., 16.]])
    scannet_results['semantic_labels'] = np.array([1, 5])

    scannet_results = scannet_sample_points(scannet_results)
    points = scannet_results.get('points', None)
    pcl_color = scannet_results.get('pcl_color', None)
    instance_labels = scannet_results.get('instance_labels', None)
    semantic_labels = scannet_results.get('semantic_labels', None)
    assert points.shape == (1, 4)
    assert pcl_color.shape == (1, 3)
    assert instance_labels.shape == (1, )
    assert semantic_labels.shape == (1, )

    sunrgbd_sample_points = PointSample('sunrgbd', 1)
    sunrgbd_results = dict()
    sunrgbd_results['points'] = np.array(
        [[1.2113925, 2.8755326, -1.1801991, 0.01056887],
         [3.6554186, 4.5093756, 0.33279705, 1.523565]])

    sunrgbd_results = sunrgbd_sample_points(sunrgbd_results)
    point_cloud = sunrgbd_results.get('points', None)
    assert point_cloud.shape == (1, 4)


test_indoor_sample()
