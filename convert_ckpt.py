# from mmengine.config import Config
# cfg = Config.fromfile(path)
# print(cfg)
import torch

mm3d_model = torch.load('checkpoints/dsvt_convert.pth')
dsvt_model = dict()
dsvt_model['model_state'] = dict()
for k, v in mm3d_model.items():
    if 'voxel_encoder' in k:
        k = k.replace('voxel_encoder', 'vfe')
    if 'middle_encoder' in k:
        k = k.replace('middle_encoder', 'backbone_3d')
    if 'backbone.' in k:
        k = k.replace('backbone', 'backbone_2d')
    if 'neck' in k:
        k = k.replace('neck', 'backbone_2d')
    if 'bbox_head.shared_conv' in k:
        k = k.replace('bbox_head.shared_conv.conv', 'dense_head.shared_conv.0')
        k = k.replace('bbox_head.shared_conv.bn', 'dense_head.shared_conv.1')
    if 'bbox_head.task_heads' in k:
        k = k.replace('bbox_head.task_heads', 'dense_head.heads_list')
        if 'reg' in k:
            k = k.replace('reg', 'center')
        if 'height' in k:
            k = k.replace('height', 'center_z')
        if 'heatmap' in k:
            k = k.replace('heatmap', 'hm')
        if '0.conv' in k:
            k = k.replace('0.conv', '0.0')
        if '0.bn' in k:
            k = k.replace('0.bn', '0.1')
    dsvt_model['model_state'][k] = v
torch.save(dsvt_model, 'dsvt_ckpt.pth')
