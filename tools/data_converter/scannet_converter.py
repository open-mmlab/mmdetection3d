import pickle
from pathlib import Path

from tools.data_converter.scannet_data_utils import ScannetObject


def create_scannet_info_file(data_path,
                             pkl_prefix='scannet_',
                             save_path=None,
                             relative_path=True):
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    train_filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    val_filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    dataset = ScannetObject(root_path=data_path, split='train')
    train_split, val_split = 'train', 'val'

    dataset.set_split(train_split)
    scannet_infos_train = dataset.get_scannet_infos(has_label=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(scannet_infos_train, f)
    print('Scannet info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    scannet_infos_val = dataset.get_scannet_infos(has_label=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(scannet_infos_val, f)
    print('Scannet info val file is saved to %s' % val_filename)


if __name__ == '__main__':
    create_scannet_info_file(
        data_path='./data/scannet', save_path='./data/scannet')
