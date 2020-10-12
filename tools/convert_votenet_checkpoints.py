import argparse
import torch
from mmcv import Config
from mmcv.runner import load_state_dict

from mmdet3d.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='path of the output checkpoint file')
    parser.add_argument(
        '--model',
        choices=['sunrgbd', 'scannet'],
        default='sunrgbd',
        help='type of the model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.model == 'scannet':
        NUM_CLASSES = 18
    else:
        NUM_CLASSES = 10

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

    RENAME_KEYS = {
        'bbox_head.conv_pred.shared_convs.layer0.conv.weight':
        'bbox_head.conv_pred.0.conv.weight',
        'bbox_head.conv_pred.shared_convs.layer0.conv.bias':
        'bbox_head.conv_pred.0.conv.bias',
        'bbox_head.conv_pred.shared_convs.layer0.bn.weight':
        'bbox_head.conv_pred.0.bn.weight',
        'bbox_head.conv_pred.shared_convs.layer0.bn.bias':
        'bbox_head.conv_pred.0.bn.bias',
        'bbox_head.conv_pred.shared_convs.layer0.bn.running_mean':
        'bbox_head.conv_pred.0.bn.running_mean',
        'bbox_head.conv_pred.shared_convs.layer0.bn.running_var':
        'bbox_head.conv_pred.0.bn.running_var',
        'bbox_head.conv_pred.shared_convs.layer1.conv.weight':
        'bbox_head.conv_pred.1.conv.weight',
        'bbox_head.conv_pred.shared_convs.layer1.conv.bias':
        'bbox_head.conv_pred.1.conv.bias',
        'bbox_head.conv_pred.shared_convs.layer1.bn.weight':
        'bbox_head.conv_pred.1.bn.weight',
        'bbox_head.conv_pred.shared_convs.layer1.bn.bias':
        'bbox_head.conv_pred.1.bn.bias',
        'bbox_head.conv_pred.shared_convs.layer1.bn.running_mean':
        'bbox_head.conv_pred.1.bn.running_mean',
        'bbox_head.conv_pred.shared_convs.layer1.bn.running_var':
        'bbox_head.conv_pred.1.bn.running_var'
    }

    DEL_KEYS = [
        'bbox_head.conv_pred.0.bn.num_batches_tracked',
        'bbox_head.conv_pred.1.bn.num_batches_tracked'
    ]

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = torch.load(args.checkpoint)
    orig_ckpt = checkpoint['state_dict']
    converted_ckpt = orig_ckpt.copy()

    for new_key, old_key in RENAME_KEYS.items():
        converted_ckpt[new_key] = converted_ckpt.pop(old_key)

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

    for key in DEL_KEYS:
        converted_ckpt.pop(key)

    # check the converted checkpoint by loading to the model
    load_state_dict(model, converted_ckpt, strict=True)
    checkpoint['state_dict'] = converted_ckpt
    torch.save(checkpoint, args.out)


if __name__ == '__main__':
    main()
