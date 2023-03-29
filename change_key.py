from collections import OrderedDict

import torch

ckpt = torch.load('checkpoints/tpv10_lidarseg_v2.pth')
new_ckpt = OrderedDict()
for k, v in ckpt.items():
    if k.startswith('module.img_backbone'):
        new_ckpt[k.replace('module.img_backbone', 'backbone')] = v
    elif k.startswith('module.img_neck'):
        new_ckpt[k.replace('module.img_neck', 'neck')] = v
    elif k.startswith('module.tpv_head.positional_encoding'):
        new_ckpt[k.replace('module.tpv_head.positional_encoding',
                           'encoder.positional_encoding')] = v
    elif k.startswith('module.tpv_head.encoder'):
        new_ckpt[k.replace('module.tpv_head.encoder', 'encoder')] = v
    elif k.startswith('module.tpv_aggregator'):
        new_ckpt[k.replace('module.tpv_aggregator', 'decode_head')] = v
    elif k.startswith('module.tpv_head.level_embeds'):
        new_ckpt[k.replace('module.tpv_head.level_embeds',
                           'encoder.level_embeds')] = v
    elif k.startswith('module.tpv_head.cams_embeds'):
        new_ckpt[k.replace('module.tpv_head.cams_embeds',
                           'encoder.cams_embeds')] = v
    elif k.startswith('module.tpv_head.tpv_embedding'):
        new_ckpt[k.replace('module.tpv_head.tpv_embedding',
                           'encoder.tpv_embedding')] = v

torch.save(new_ckpt, 'checkpoints/tpvformer.pth')
