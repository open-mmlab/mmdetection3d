import os

import mmcv
from tools.data_converter.sunrgbd_data_utils import SUNRGBDData


def create_sunrgbd_info_file(data_path,
                             pkl_prefix='sunrgbd',
                             save_path=None,
                             use_v1=False):
    '''
    Create sunrgbd information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str): Prefix ofr the pkl to be saved. Default: 'sunrgbd'.
        save_path (str): Path of the pkl to be saved. Default: None.
        use_v1 (bool): Whether to use v1. Default: False.

    Returns:
        None

    '''
    assert os.path.exists(data_path)
    if save_path is None:
        save_path = data_path
    else:
        save_path = save_path
    assert os.path.exists(save_path)
    train_filename = os.path.join(save_path, f'{pkl_prefix}_infos_train.pkl')
    val_filename = os.path.join(save_path, f'{pkl_prefix}_infos_val.pkl')
    train_dataset = SUNRGBDData(
        root_path=data_path, split='train', use_v1=use_v1)
    val_dataset = SUNRGBDData(root_path=data_path, split='val', use_v1=use_v1)
    sunrgbd_infos_train = train_dataset.get_sunrgbd_infos(has_label=True)
    with open(train_filename, 'wb') as f:
        mmcv.dump(sunrgbd_infos_train, f, 'pkl')
    print(f'Sunrgbd info train file is saved to {train_filename}')
    sunrgbd_infos_val = val_dataset.get_sunrgbd_infos(has_label=True)
    with open(val_filename, 'wb') as f:
        mmcv.dump(sunrgbd_infos_val, f, 'pkl')
    print(f'Sunrgbd info val file is saved to {val_filename}')
