import mmcv
import numpy as np
import torch

from mmdet3d.models.fusion_layers import VoteFusion

snap = mmcv.load('./work_dirs/snap.pkl')

fusion = VoteFusion(img_norm_cfg={'mean': []})
calibs = {
    'Rt': torch.from_numpy(snap['calib_rt']).cuda().float(),
    'K': torch.from_numpy(snap['calib_k']).cuda().float()
}
imgs = torch.from_numpy(snap['img']).cuda()
bboxes_2d_rescaled = torch.from_numpy(snap['bbox_2d']).cuda()[:, :10]
seeds_3d = torch.from_numpy(snap['seeds_3d']).cuda()
img_metas = {}
for k in snap.keys():
    if type(snap[k]) == np.ndarray:
        img_metas.update({k: torch.from_numpy(snap[k]).cuda()})
    else:
        img_metas.update({k: snap[k]})

print(imgs.shape, bboxes_2d_rescaled.shape, seeds_3d.shape)
# print(img_metas)
print(calibs)
out = fusion(imgs, bboxes_2d_rescaled, seeds_3d, [img_metas], calibs)
print(out)
print(snap.keys())
