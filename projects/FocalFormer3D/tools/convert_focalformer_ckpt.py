#!/usr/bin/env python3
"""Convert FocalFormer3D checkpoint from old mmcv/mmdet2.x key format to new
mmdet3.x/mmdet3d 1.4 key format.

Key remapping:
  attentions.0  ->  self_attn
  attentions.1  ->  cross_attn
  ffns.0        ->  ffn

Usage:
  python convert_focalformer_ckpt.py \
      --src /path/to/FocalFormer3D_L_ep6_mAP664_NDS709.pth \
      --dst /path/to/FocalFormer3D_L_ep6_converted.pth
"""
import argparse

import torch


def convert_checkpoint(src_path, dst_path):
    ckpt = torch.load(src_path, map_location='cpu', weights_only=False)

    if 'state_dict' not in ckpt:
        return

    old_state_dict = ckpt['state_dict']
    new_state_dict = {}
    remapped = 0

    for key, value in old_state_dict.items():
        new_key = key

        if 'pts_bbox_head.decoder.' in key:
            new_key = new_key.replace('.attentions.0.', '.self_attn.')
            new_key = new_key.replace('.attentions.1.', '.cross_attn.')
            new_key = new_key.replace('.ffns.0.', '.ffn.')

            if new_key != key:
                remapped += 1

        new_state_dict[new_key] = value

    # Show a few examples
    count = 0
    for old_key in old_state_dict:
        new_key = old_key
        new_key = new_key.replace('.attentions.0.', '.self_attn.')
        new_key = new_key.replace('.attentions.1.', '.cross_attn.')
        new_key = new_key.replace('.ffns.0.', '.ffn.')
        if new_key != old_key:
            count += 1
            if count >= 3:
                break

    ckpt['state_dict'] = new_state_dict
    torch.save(ckpt, dst_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Source checkpoint path')
    parser.add_argument(
        '--dst', required=True, help='Destination checkpoint path')
    args = parser.parse_args()
    convert_checkpoint(args.src, args.dst)
