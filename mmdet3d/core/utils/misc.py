from functools import partial

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from six.moves import map, zip


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


def merge_batch(data):
    for key, elems in data.items():
        if key in ['voxels', 'num_points', 'voxel_labels', 'voxel_centers']:
            data[key]._data[0] = torch.cat(elems._data[0], dim=0)
        elif key == 'coors':
            coors = []
            for i, coor in enumerate(elems._data[0]):
                coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
                coors.append(coor_pad)
            data[key]._data[0] = torch.cat(coors, dim=0)
    return data


def merge_hook_batch(data):
    for key, elems in data.items():
        if key in ['voxels', 'num_points', 'voxel_labels', 'voxel_centers']:
            data[key] = torch.cat(elems, dim=0)
        elif key == 'coors':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
                coors.append(coor_pad)
            data[key] = torch.cat(coors, dim=0)
    return data
