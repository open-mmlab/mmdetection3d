# Copyright (c) OpenMMLab. All rights reserved.
import itertools

from mmcv.cnn.bricks.registry import CONV_LAYERS
from torch.nn.parameter import Parameter


def register_spconv2():
    """This func registers spconv2.0 spconv ops to overwrite the default mmcv
    spconv ops."""
    try:
        from spconv.pytorch import (SparseConv2d, SparseConv3d, SparseConv4d,
                                    SparseConvTranspose2d,
                                    SparseConvTranspose3d, SparseInverseConv2d,
                                    SparseInverseConv3d, SparseModule,
                                    SubMConv2d, SubMConv3d, SubMConv4d)
    except ImportError:
        return False
    else:
        CONV_LAYERS._register_module(SparseConv2d, 'SparseConv2d', force=True)
        CONV_LAYERS._register_module(SparseConv3d, 'SparseConv3d', force=True)
        CONV_LAYERS._register_module(SparseConv4d, 'SparseConv4d', force=True)

        CONV_LAYERS._register_module(
            SparseConvTranspose2d, 'SparseConvTranspose2d', force=True)
        CONV_LAYERS._register_module(
            SparseConvTranspose3d, 'SparseConvTranspose3d', force=True)

        CONV_LAYERS._register_module(
            SparseInverseConv2d, 'SparseInverseConv2d', force=True)
        CONV_LAYERS._register_module(
            SparseInverseConv3d, 'SparseInverseConv3d', force=True)

        CONV_LAYERS._register_module(SubMConv2d, 'SubMConv2d', force=True)
        CONV_LAYERS._register_module(SubMConv3d, 'SubMConv3d', force=True)
        CONV_LAYERS._register_module(SubMConv4d, 'SubMConv4d', force=True)
        SparseModule._version = 2
        SparseModule._load_from_state_dict = _load_from_state_dict
        return True


def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs):
    """Rewrite this func to compat the convolutional kernel weights between
    spconv 1.x in MMCV and 2.x in spconv2.x.

    Kernel weights in MMCV spconv has shape in (D,H,W,in_channel,out_channel) ,
    while those in spcon2.x is in (out_channel,D,H,W,in_channel).
    """
    version = local_metadata.get('version', None)
    for hook in self._load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs)

    local_name_params = itertools.chain(self._parameters.items(),
                                        self._buffers.items())
    local_state = {k: v.data for k, v in local_name_params if v is not None}

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]

            # Backward compatibility: loading 1-dim tensor from
            # 0.3.* to version 0.4+
            if len(param.shape) == 0 and len(input_param.shape) == 1:
                input_param = input_param[0]
            if version != 2:
                dims = [len(input_param.shape) - 1] + list(
                    range(len(input_param.shape) - 1))
                input_param = input_param.permute(*dims)
            if input_param.shape != param.shape:
                # local shape should match the one in checkpoint
                error_msgs.append(
                    f'size mismatch for {key}: copying a param with '
                    f'shape {key, input_param.shape} from checkpoint,'
                    f'the shape in current model is {param.shape}.')
                continue

            if isinstance(input_param, Parameter):
                # backwards compatibility for serialized parameters
                input_param = input_param.data
            try:
                param.copy_(input_param)
            except Exception:
                error_msgs.append(
                    f'While copying the parameter named "{key}", whose '
                    f'dimensions in the model are {param.size()} and whose '
                    f'dimensions in the checkpoint are {input_param.size()}.')
        elif strict:
            missing_keys.append(key)

    if strict:
        for key, input_param in state_dict.items():
            if key.startswith(prefix):
                input_name = key[len(prefix):]
                input_name = input_name.split(
                    '.', 1)[0]  # get the name of param/buffer/child
                if input_name not in self._modules \
                        and input_name not in local_state:
                    unexpected_keys.append(key)
