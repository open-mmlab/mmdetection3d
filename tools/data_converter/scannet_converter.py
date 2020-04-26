import os
import pickle
from pathlib import Path

from tools.data_converter.scannet_data_utils import ScannetObject


def create_scannet_info_file(data_path, pkl_prefix='scannet', save_path=None):
    assert os.path.exists(data_path)
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    assert os.path.exists(save_path)
    train_filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    val_filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    train_dataset = ScannetObject(root_path=data_path, split='train')
    val_dataset = ScannetObject(root_path=data_path, split='val')
    scannet_infos_train = train_dataset.get_scannet_infos(has_label=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(scannet_infos_train, f)
    print('Scannet info train file is saved to %s' % train_filename)
    scannet_infos_val = val_dataset.get_scannet_infos(has_label=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(scannet_infos_val, f)
    print('Scannet info val file is saved to %s' % val_filename)


if __name__ == '__main__':
    create_scannet_info_file(
        data_path='./data/scannet', save_path='./data/scannet')
