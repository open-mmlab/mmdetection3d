import os

import mmcv
from tools.data_converter.scannet_data_utils import ScanNetData


def create_scannet_info_file(data_path, pkl_prefix='scannet', save_path=None):
    '''
        Create scannet information file.

        Get information of the raw data and save it to the pkl file.

        Args:
            data_path (str): Path of the data.
            pkl_prefix (str): Prefix ofr the pkl to be saved. Default: 'scannet'. # noqa: E501
            save_path (str): Path of the pkl to be saved. Default: None.

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
    train_dataset = ScanNetData(root_path=data_path, split='train')
    val_dataset = ScanNetData(root_path=data_path, split='val')
    scannet_infos_train = train_dataset.get_scannet_infos(has_label=True)
    with open(train_filename, 'wb') as f:
        mmcv.dump(scannet_infos_train, f, 'pkl')
    print(f'Scannet info train file is saved to {train_filename}')
    scannet_infos_val = val_dataset.get_scannet_infos(has_label=True)
    with open(val_filename, 'wb') as f:
        mmcv.dump(scannet_infos_val, f, 'pkl')
    print(f'Scannet info val file is saved to {val_filename}')
