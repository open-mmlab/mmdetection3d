import mmcv

from mmdet.core.evaluation import dataset_aliases


def kitti_classes():
    return [
        'Car',
        'Pedestrian',
        'Cyclist',
        'Van',
        'Person_sitting',
    ]


dataset_aliases.update({'kitti': ['KITTI', 'kitti']})


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels
