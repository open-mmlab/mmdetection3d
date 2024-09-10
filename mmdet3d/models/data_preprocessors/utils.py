# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def multiview_img_stack_batch(tensor_list: List[Tensor],
                              pad_size_divisor: int = 1,
                              pad_value: Union[int, float] = 0) -> Tensor:
    """Compared to the ``stack_batch`` in `mmengine.model.utils`,
    multiview_img_stack_batch further handle the multiview images.

    See diff of padded_sizes[:, :-2] = 0 vs padded_sizes[:, 0] = 0 in line 47.

    Stack multiple tensors to form a batch and pad the tensor to the max shape
    use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding to
            ensure the shape of each dim is divisible by ``pad_size_divisor``.
            This depends on the model, and many models need to be divisible by
            32. Defaults to 1.
        pad_value (int or float): The padding value. Defaults to 0.

    Returns:
        Tensor: The n dim tensor.
    """
    assert isinstance(tensor_list, list), \
        f'Expected input type to be list, but got {type(tensor_list)}'
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({tensor.ndim for tensor in tensor_list}) == 1, \
        'Expected the dimensions of all tensors must be the same, ' \
        f'but got {[tensor.ndim for tensor in tensor_list]}'

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor(
        [tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(
        torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel, which should not be padded.
    padded_sizes[:, :-2] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)
