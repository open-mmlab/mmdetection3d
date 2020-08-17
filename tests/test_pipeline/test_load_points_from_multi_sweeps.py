import numpy as np

from mmdet3d.datasets.pipelines.loading import LoadPointsFromMultiSweeps


def test_load_points_from_multi_sweeps():
    np.random.seed(0)
    file_client_args = dict(backend='disk')
    load_points_from_multi_sweeps = LoadPointsFromMultiSweeps(
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True)
    points = np.random.random([100, 5]) * 2
    results = dict(points=points, sweeps=[], timestamp=None)
    results = load_points_from_multi_sweeps(results)
    assert results['points'].shape == (775, 5)

    sensor2lidar_rotation = np.array(
        [[9.99999967e-01, 1.13183067e-05, 2.56845368e-04],
         [-1.12839618e-05, 9.99999991e-01, -1.33719456e-04],
         [-2.56846879e-04, 1.33716553e-04, 9.99999958e-01]])
    sensor2lidar_translation = np.array([-0.0009198, -0.03964854, -0.00190136])
    sweep = dict(
        data_path='tests/data/nuscenes/sweeps/LIDAR_TOP/'
        'n008-2018-09-18-12-07-26-0400__LIDAR_TOP__'
        '1537287083900561.pcd.bin',
        sensor2lidar_rotation=sensor2lidar_rotation,
        sensor2lidar_translation=sensor2lidar_translation,
        timestamp=0)
    results = dict(points=points, sweeps=[sweep], timestamp=1.0)

    results = load_points_from_multi_sweeps(results)
    assert results['points'].shape == (451, 5)

    results = dict(points=points, sweeps=[sweep] * 10, timestamp=1.0)
    results = load_points_from_multi_sweeps(results)
    assert results['points'].shape == (3259, 5)
