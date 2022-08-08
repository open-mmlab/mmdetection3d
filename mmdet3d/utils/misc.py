# Copyright (c) OpenMMLab. All rights reserved.


def replace_ceph_backend(cfg):
    cfg_pretty_text = cfg.pretty_text

    replace_strs = r'''file_client_args = dict(
                    backend='petrel',
                    path_mapping=dict({
                        '.data/INPLACEHOLD/':
                        's3://openmmlab/datasets/detection3d/INPLACEHOLD/',
                        'data/INPLACEHOLD/':
                        's3://openmmlab/datasets/detection3d/INPLACEHOLD/'
                    }))
                '''

    if 'nuscenes' in cfg_pretty_text:
        replace_strs = replace_strs.replace('INPLACEHOLD', 'nuscenes')
    elif 'lyft' in cfg_pretty_text:
        replace_strs = replace_strs.replace('INPLACEHOLD', 'lyft')
    elif 'kitti' in cfg_pretty_text:
        replace_strs = replace_strs.replace('INPLACEHOLD', 'kitti')
    elif 'waymo' in cfg_pretty_text:
        replace_strs = replace_strs.replace('INPLACEHOLD', 'waymo')
    elif 'scannet' in cfg_pretty_text:
        replace_strs = replace_strs.replace('INPLACEHOLD', 'scannet_processed')
    elif 's3dis' in cfg_pretty_text:
        replace_strs = replace_strs.replace('INPLACEHOLD', 's3dis_processed')
    elif 'sunrgbd' in cfg_pretty_text:
        replace_strs = replace_strs.replace('INPLACEHOLD', 'sunrgbd')
    elif 'semantickitti' in cfg_pretty_text:
        replace_strs = replace_strs.replace('INPLACEHOLD', 'semantickitti')
    elif 'nuimages' in cfg_pretty_text:
        replace_strs = replace_strs.replace('INPLACEHOLD', 'nuimages')
    else:
        NotImplemented('Does not support global replacement')

    replace_strs = replace_strs.replace(' ', '').replace('\n', '')

    # use data info file from ceph
    # cfg_pretty_text = cfg_pretty_text.replace(
    #   'ann_file', replace_strs + ', ann_file')

    # replace LoadImageFromFile
    replace_strs = replace_strs.replace(' ', '').replace('\n', '')
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadImageFromFile\'', 'LoadImageFromFile\',' + replace_strs)

    # replace LoadImageFromFileMono3D
    cfg_pretty_text = cfg_pretty_text.replace(
        'LoadImageFromFileMono3D\'',
        'LoadImageFromFileMono3D\',' + replace_strs)

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

    # replace dbsampler
    cfg_pretty_text = cfg_pretty_text.replace('info_path',
                                              replace_strs + ', info_path')

    cfg = cfg.fromstring(cfg_pretty_text, file_format='.py')
    return cfg
