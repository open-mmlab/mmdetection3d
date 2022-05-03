# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

# yapf: disable
from mmdet3d.datasets.pipelines import (Collect3D, DefaultFormatBundle3D,
                                        LoadAnnotations3D,
                                        LoadImageFromFileMono3D,
                                        LoadMultiViewImageFromFiles,
                                        LoadPointsFromFile,
                                        LoadPointsFromMultiSweeps,
                                        MultiScaleFlipAug3D,
                                        PointSegClassMapping)
from mmdet.datasets.pipelines import LoadImageFromFile, MultiScaleFlipAug
# yapf: enable
from .builder import PIPELINES


def is_loading_function(transform):
    """Judge whether a transform function is a loading function.

    Note: `MultiScaleFlipAug3D` is a wrapper for multiple pipeline functions,
    so we need to search if its inner transforms contain any loading function.

    Args:
        transform (dict | :obj:`Pipeline`): A transform config or a function.

    Returns:
        bool: Whether it is a loading function. None means can't judge.
            When transform is `MultiScaleFlipAug3D`, we return None.
    """
    # TODO: use more elegant way to distinguish loading modules
    loading_functions = (LoadImageFromFile, LoadPointsFromFile,
                         LoadAnnotations3D, LoadMultiViewImageFromFiles,
                         LoadPointsFromMultiSweeps, DefaultFormatBundle3D,
                         Collect3D, LoadImageFromFileMono3D,
                         PointSegClassMapping)
    if isinstance(transform, dict):
        obj_cls = PIPELINES.get(transform['type'])
        if obj_cls is None:
            return False
        if obj_cls in loading_functions:
            return True
        if obj_cls in (MultiScaleFlipAug3D, MultiScaleFlipAug):
            return None
    elif callable(transform):
        if isinstance(transform, loading_functions):
            return True
        if isinstance(transform, (MultiScaleFlipAug3D, MultiScaleFlipAug)):
            return None
    return False


def get_loading_pipeline(pipeline):
    """Only keep loading image, points and annotations related configuration.

    Args:
        pipeline (list[dict] | list[:obj:`Pipeline`]):
            Data pipeline configs or list of pipeline functions.

    Returns:
        list[dict] | list[:obj:`Pipeline`]): The new pipeline list with only
            keep loading image, points and annotations related configuration.

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
        >>> assert expected_pipelines == \
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline = []
    for transform in pipeline:
        is_loading = is_loading_function(transform)
        if is_loading is None:  # MultiScaleFlipAug3D
            # extract its inner pipeline
            if isinstance(transform, dict):
                inner_pipeline = transform.get('transforms', [])
            else:
                inner_pipeline = transform.transforms.transforms
            loading_pipeline.extend(get_loading_pipeline(inner_pipeline))
        elif is_loading:
            loading_pipeline.append(transform)
    assert len(loading_pipeline) > 0, \
        'The data pipeline in your config file must include ' \
        'loading step.'
    return loading_pipeline


def extract_result_dict(results, key):
    """Extract and return the data corresponding to key in result dict.

    ``results`` is a dict output from `pipeline(input_dict)`, which is the
        loaded data from ``Dataset`` class.
    The data terms inside may be wrapped in list, tuple and DataContainer, so
        this function essentially extracts data from these wrappers.

    Args:
        results (dict): Data loaded using pipeline.
        key (str): Key of the desired data.

    Returns:
        np.ndarray | torch.Tensor: Data term.
    """
    if key not in results.keys():
        return None
    # results[key] may be data or list[data] or tuple[data]
    # data may be wrapped inside DataContainer
    data = results[key]
    if isinstance(data, (list, tuple)):
        data = data[0]
    if isinstance(data, mmcv.parallel.DataContainer):
        data = data._data
    return data
