import os
import pickle
from pathlib import Path

from tools.data_converter.sunrgbd_data_utils import SUNRGBDObject


def create_sunrgbd_info_file(data_path,
                             pkl_prefix='sunrgbd',
                             save_path=None,
                             use_v1=False):
    assert os.path.exists(data_path)
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    assert os.path.exists(save_path)
    train_filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    val_filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    train_dataset = SUNRGBDObject(
        root_path=data_path, split='train', use_v1=use_v1)
    val_dataset = SUNRGBDObject(
        root_path=data_path, split='val', use_v1=use_v1)
    sunrgbd_infos_train = train_dataset.get_sunrgbd_infos(has_label=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(sunrgbd_infos_train, f)
    print('Sunrgbd info train file is saved to %s' % train_filename)
    sunrgbd_infos_val = val_dataset.get_sunrgbd_infos(has_label=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(sunrgbd_infos_val, f)
    print('Sunrgbd info val file is saved to %s' % val_filename)


if __name__ == '__main__':
    create_sunrgbd_info_file(
        data_path='./data/sunrgbd/sunrgbd_trainval',
        save_path='./data/sunrgbd')
