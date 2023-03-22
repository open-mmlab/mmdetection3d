from os import path as osp
from pathlib import Path

import mmengine

total_num = {
    0: 4541,
    1: 1101,
    2: 4661,
    3: 801,
    4: 271,
    5: 2761,
    6: 1101,
    7: 1101,
    8: 4071,
    9: 1591,
    10: 1201,
    11: 921,
    12: 1061,
    13: 3281,
    14: 631,
    15: 1901,
    16: 1731,
    17: 491,
    18: 1801,
    19: 4981,
    20: 831,
    21: 2721,
}
fold_split = {
    'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    'val': [8],
    'test': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}
split_list = ['train', 'valid', 'test']


def get_semantickitti_info(split):
    """Create info file in the form of
        data_infos={
            'metainfo': {'DATASET': 'SemanticKITTI'},
            'data_list': {
                00000: {
                    'lidar_points':{
                        'lidat_path':'sequences/00/velodyne/000000.bin'
                    },
                    'pts_semantic_mask_path':
                        'sequences/000/labels/000000.labbel',
                    'sample_id': '00'
                },
                ...
            }
        }
    """
    data_infos = dict()
    data_infos['metainfo'] = dict(DATASET='SemanticKITTI')
    data_list = []
    for i_folder in fold_split[split]:
        for j in range(0, total_num[i_folder]):
            data_list.append({
                'lidar_points': {
                    'lidar_path':
                    osp.join('sequences',
                             str(i_folder).zfill(2), 'velodyne',
                             str(j).zfill(6) + '.bin'),
                    'num_pts_feats':
                    4
                },
                'pts_semantic_mask_path':
                osp.join('sequences',
                         str(i_folder).zfill(2), 'labels',
                         str(j).zfill(6) + '.label'),
                'sample_id':
                str(i_folder) + str(j)
            })
    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_semantickitti_info_file(pkl_prefix, save_path):
    """Create info file of SemanticKITTI dataset.

    Directly generate info file without raw data.

    Args:
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
    """
    print('Generate info.')
    save_path = Path(save_path)

    semantickitti_infos_train = get_semantickitti_info(split='train')
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'SemanticKITTI info train file is saved to {filename}')
    mmengine.dump(semantickitti_infos_train, filename)
    semantickitti_infos_val = get_semantickitti_info(split='val')
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'SemanticKITTI info val file is saved to {filename}')
    mmengine.dump(semantickitti_infos_val, filename)
    semantickitti_infos_test = get_semantickitti_info(split='test')
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'SemanticKITTI info test file is saved to {filename}')
    mmengine.dump(semantickitti_infos_test, filename)
