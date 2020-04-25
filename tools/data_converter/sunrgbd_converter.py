import pickle
from pathlib import Path

from tools.data_converter.sunrgbd_data_utils import SUNRGBDObject


def create_sunrgbd_info_file(data_path,
                             pkl_prefix='sunrgbd_',
                             save_path=None,
                             relative_path=True):
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    train_filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    val_filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    dataset = SUNRGBDObject(root_path=data_path, split='train')
    train_split, val_split = 'train', 'val'

    dataset.set_split(train_split)
    sunrgbd_infos_train = dataset.get_sunrgbd_infos(has_label=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(sunrgbd_infos_train, f)
    print('Sunrgbd info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    sunrgbd_infos_val = dataset.get_sunrgbd_infos(has_label=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(sunrgbd_infos_val, f)
    print('Sunrgbd info val file is saved to %s' % val_filename)


if __name__ == '__main__':
    create_sunrgbd_info_file(
        data_path='./data/sunrgbd/sunrgbd_trainval',
        save_path='./data/sunrgbd')
