# Copyright (c) OpenMMLab. All rights reserved.


def replace_ceph_backend(cfg):
    cfg_pretty_text = cfg.pretty_text

    replace_strs = \
        r'''backend_args = dict(
            backend='petrel',
            path_mapping=dict({
                './data/DATA/': 's3://openmmlab/datasets/detection3d/CEPH/',
                'data/DATA/': 's3://openmmlab/datasets/detection3d/CEPH/'
            }))
        '''

    if 'nuscenes' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'nuscenes')
        replace_strs = replace_strs.replace('CEPH', 'nuscenes')
    elif 'lyft' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'lyft')
        replace_strs = replace_strs.replace('CEPH', 'lyft')
    elif 'waymo' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'waymo')
        replace_strs = replace_strs.replace('CEPH', 'waymo')
    elif 'kitti' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'kitti')
        replace_strs = replace_strs.replace('CEPH', 'kitti')
    elif 'scannet' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'scannet')
        replace_strs = replace_strs.replace('CEPH', 'scannet_processed')
    elif 's3dis' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 's3dis')
        replace_strs = replace_strs.replace('CEPH', 's3dis_processed')
    elif 'sunrgbd' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'sunrgbd')
        replace_strs = replace_strs.replace('CEPH', 'sunrgbd_processed')
    elif 'semantickitti' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'semantickitti')
        replace_strs = replace_strs.replace('CEPH', 'semantickitti')
    elif 'nuimages' in cfg_pretty_text:
        replace_strs = replace_strs.replace('DATA', 'nuimages')
        replace_strs = replace_strs.replace('CEPH', 'nuimages')
    else:
        NotImplemented('Does not support global replacement')

    replace_strs = replace_strs.replace(' ', '').replace('\n', '')

    # use data info file from ceph
    # cfg_pretty_text = cfg_pretty_text.replace(
    #   'ann_file', replace_strs + ', ann_file')

    cfg_pretty_text = cfg_pretty_text.replace('backend_args=None', '')

    # replace LoadImageFromFile
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadImageFromFile\'', 'LoadImageFromFile\',' + replace_strs)

    # replace LoadImageFromFileMono3D
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadImageFromFileMono3D\'',
        'LoadImageFromFileMono3D\',' + replace_strs)

    # replace LoadMultiViewImageFromFiles
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadMultiViewImageFromFiles\'',
        'LoadMultiViewImageFromFiles\',' + replace_strs)

    # replace LoadPointsFromFile
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadPointsFromFile\'', 'LoadPointsFromFile\',' + replace_strs)

    # replace LoadPointsFromMultiSweeps
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadPointsFromMultiSweeps\'',
        'LoadPointsFromMultiSweeps\',' + replace_strs)

    # replace LoadAnnotations
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadAnnotations\'', 'LoadAnnotations\',' + replace_strs)

    # replace LoadAnnotations3D
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadAnnotations3D\'', 'LoadAnnotations3D\',' + replace_strs)

    # replace KittiMetric
    cfg_pretty_text = cfg_pretty_text.replace('KittiMetric\'',
                                              'KittiMetric\',' + replace_strs)

    # replace LyftMetric
    cfg_pretty_text = cfg_pretty_text.replace('LyftMetric\'',
                                              'LyftMetric\',' + replace_strs)

    # replace NuScenesMetric
    cfg_pretty_text = cfg_pretty_text.replace(
        'NuScenesMetric\'', 'NuScenesMetric\',' + replace_strs)

    # replace WaymoMetric
    cfg_pretty_text = cfg_pretty_text.replace('WaymoMetric\'',
                                              'WaymoMetric\',' + replace_strs)

    # replace dbsampler
    cfg_pretty_text = cfg_pretty_text.replace('info_path',
                                              replace_strs + ', info_path')

    cfg = cfg.fromstring(cfg_pretty_text, file_format='.py')
    return cfg
