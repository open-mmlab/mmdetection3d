from mmdet3d.datasets.pipelines import (Collect3D, DefaultFormatBundle3D,
                                        LoadAnnotations3D,
                                        LoadMultiViewImageFromFiles,
                                        LoadPointsFromFile,
                                        LoadPointsFromMultiSweeps)
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadImageFromFile


def get_loading_pipeline(pipeline):
    """Only keep loading image, points and annotations related configuration.

    Args:
        pipeline (list[dict]): Data pipeline configs.

    Returns:
        list[dict]: The new pipeline list with only keep
            loading image, points and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='Resize',
        ...         img_scale=[(640, 192), (2560, 768)], keep_ratio=True),
        ...    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        ...    dict(type='PointsRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='ObjectRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='PointShuffle'),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> assert expected_pipelines ==\
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline_cfg = []
    for cfg in pipeline:
        obj_cls = PIPELINES.get(cfg['type'])
        # TODO: use more elegant way to distinguish loading modules
        if obj_cls is not None and obj_cls in (
                LoadImageFromFile, LoadPointsFromFile, LoadAnnotations3D,
                LoadMultiViewImageFromFiles, LoadPointsFromMultiSweeps,
                DefaultFormatBundle3D, Collect3D):
            loading_pipeline_cfg.append(cfg)
    assert len(loading_pipeline_cfg) > 0, \
        'The data pipeline in your config file must include ' \
        'loading step.'
    return loading_pipeline_cfg
