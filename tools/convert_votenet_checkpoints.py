import argparse
import tempfile
import torch
from mmcv import Config
from mmcv.runner import load_state_dict

from mmdet3d.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D upgrade model version(before v0.6.0) of VoteNet')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='path of the output checkpoint file')
    args = parser.parse_args()
    return args


def parse_config(config_strings):
    """Parse config from strings.

    Args:
        config_strings (string): strings of model config.

    Returns:
        Config: model config
    """
    temp_file = tempfile.NamedTemporaryFile()
    config_path = f'{temp_file.name}.py'
    with open(config_path, 'w') as f:
        f.write(config_strings)

    config = Config.fromfile(config_path)

    # Update backbone config
    if 'pool_mod' in config.model.backbone:
        config.model.backbone.pop('pool_mod')

    if 'sa_cfg' not in config.model.backbone:
        config.model.backbone['sa_cfg'] = dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)

    if 'type' not in config.model.bbox_head.vote_aggregation_cfg:
        config.model.bbox_head.vote_aggregation_cfg['type'] = 'PointSAModule'

    # Update bbox_head config
    if 'pred_layer_cfg' not in config.model.bbox_head:
        config.model.bbox_head['pred_layer_cfg'] = dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True)

    if 'feat_channels' in config.model.bbox_head:
        config.model.bbox_head.pop('feat_channels')

    if 'vote_moudule_cfg' in config.model.bbox_head:
        config.model.bbox_head['vote_module_cfg'] = config.model.bbox_head.pop(
            'vote_moudule_cfg')

    if config.model.bbox_head.vote_aggregation_cfg.use_xyz:
        config.model.bbox_head.vote_aggregation_cfg.mlp_channels[0] -= 3

    temp_file.close()

    return config


def main():
    """Convert keys in checkpoints for VoteNet.

    There can be some breaking changes during the development of mmdetection3d,
    and this tool is used for upgrading checkpoints trained with old versions
    (before v0.6.0) to the latest one.
    """
    args = parse_args()
    checkpoint = torch.load(args.checkpoint)
    cfg = parse_config(checkpoint['meta']['config'])
    # Build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    orig_ckpt = checkpoint['state_dict']
    converted_ckpt = orig_ckpt.copy()

    if cfg['dataset_type'] == 'ScanNetDataset':
        NUM_CLASSES = 18
    elif cfg['dataset_type'] == 'SUNRGBDDataset':
        NUM_CLASSES = 10
    else:
        raise NotImplementedError

    RENAME_PREFIX = {
        'bbox_head.conv_pred.0': 'bbox_head.conv_pred.shared_convs.layer0',
        'bbox_head.conv_pred.1': 'bbox_head.conv_pred.shared_convs.layer1'
    }

    DEL_KEYS = [
        'bbox_head.conv_pred.0.bn.num_batches_tracked',
        'bbox_head.conv_pred.1.bn.num_batches_tracked'
    ]

    EXTRACT_KEYS = {
        'bbox_head.conv_pred.conv_cls.weight':
        ('bbox_head.conv_pred.conv_out.weight', [(0, 2), (-NUM_CLASSES, -1)]),
        'bbox_head.conv_pred.conv_cls.bias':
        ('bbox_head.conv_pred.conv_out.bias', [(0, 2), (-NUM_CLASSES, -1)]),
        'bbox_head.conv_pred.conv_reg.weight':
        ('bbox_head.conv_pred.conv_out.weight', [(2, -NUM_CLASSES)]),
        'bbox_head.conv_pred.conv_reg.bias':
        ('bbox_head.conv_pred.conv_out.bias', [(2, -NUM_CLASSES)])
    }

    # Delete some useless keys
    for key in DEL_KEYS:
        converted_ckpt.pop(key)

    # Rename keys with specific prefix
    RENAME_KEYS = dict()
    for old_key in converted_ckpt.keys():
        for rename_prefix in RENAME_PREFIX.keys():
            if rename_prefix in old_key:
                new_key = old_key.replace(rename_prefix,
                                          RENAME_PREFIX[rename_prefix])
                RENAME_KEYS[new_key] = old_key
    for new_key, old_key in RENAME_KEYS.items():
        converted_ckpt[new_key] = converted_ckpt.pop(old_key)

    # Extract weights and rename the keys
    for new_key, (old_key, indices) in EXTRACT_KEYS.items():
        cur_layers = orig_ckpt[old_key]
        converted_layers = []
        for (start, end) in indices:
            if end != -1:
                converted_layers.append(cur_layers[start:end])
            else:
                converted_layers.append(cur_layers[start:])
        converted_layers = torch.cat(converted_layers, 0)
        converted_ckpt[new_key] = converted_layers
        if old_key in converted_ckpt.keys():
            converted_ckpt.pop(old_key)

    # Check the converted checkpoint by loading to the model
    load_state_dict(model, converted_ckpt, strict=True)
    checkpoint['state_dict'] = converted_ckpt
    torch.save(checkpoint, args.out)


if __name__ == '__main__':
    main()
