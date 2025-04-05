# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path

import mmengine
import numpy as np



def get_lidar_index_str(lidar_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{}'.format(lidar_idx)
    else:
        return '{}'.format(lidar_idx)


def get_custom_info_path(idx,
                        prefix,
                        info_type='velodyne',
                        file_tail='.bin',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    lidar_idx_str = get_lidar_index_str(idx, use_prefix_id)
    lidar_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / lidar_idx_str
    else:
        file_path = Path('testing') / info_type / lidar_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)

def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='label_2',
                   use_prefix_id=False):
    return get_custom_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)

def get_lidar_path(idx,
                  prefix,
                  training=True,
                  relative_path=True,
                  exist_check=True,
                  use_prefix_id=False):
    return get_custom_info_path(idx, prefix, 'velodyne', '.bin', training,
                                relative_path, exist_check, use_prefix_id)

def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x for x in content])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0],))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def get_custom_velodyne_info(path,
                         training=True,
                         label_info=True,
                         velodyne=True,
                         point_ids=7481,
                         num_worker=8,
                         relative_path=True,
                         with_velodyneshape=True):
    """
    custom annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for custom]velodyne: {
            lidar_idx: ...
            velodyne_path: ...
            velodyne_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: custom difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)


    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}

        # pc_info = {'lidar_idx': idx}
        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_lidar_path(
                idx, path, training, relative_path)
            pc_info['lidar_idx'] = idx

        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['point_cloud'] = pc_info


        if annotations is not None:
            info['annos'] = annotations
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        velodyne_infos = executor.map(map_func, point_ids)

    return list(velodyne_infos)

def custom_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        lidar_idx = anno['metadata']['lidar_idx']
        label_lines = []
        for j in range(anno['bbox'].shape[0]):
            label_dict = {
                'name': anno['name'][j],
                'alpha': anno['alpha'][j],
                'bbox': anno['bbox'][j],
                'location': anno['location'][j],
                'dimensions': anno['dimensions'][j],
                'rotation_y': anno['rotation_y'][j],
                'score': anno['score'][j],
            }
            label_line = custom_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / f'{get_lidar_index_str(lidar_idx)}.txt'
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)


def custom_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)
