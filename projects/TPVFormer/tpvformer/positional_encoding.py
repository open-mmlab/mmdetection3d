import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule

from mmdet.utils import MultiConfig, OptMultiConfig

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LearnedPositionalEncoding(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Defaults to 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Defaults to 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_feats: int,
        row_num_embed: int = 50,
        col_num_embed: int = 50,
        init_cfg: MultiConfig = dict(type='Uniform', layer='Embedding')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask: Tensor) -> Tensor:
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x = x.cuda()
        y = y.cuda()
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat((
                x_embed.unsqueeze(0).repeat(h, 1, 1), 
                y_embed.unsqueeze(1).repeat(1, w, 1)),
                dim=-1,
            ).permute(2, 0, 1).unsqueeze(0).repeat(
                mask.shape[0], 1, 1, 1)
        return pos


@MODELS.register_module()
class TPVFormerPositionalEncoding(BaseModule):

    def __init__(self,
                 num_feats,
                 h,
                 w,
                 z,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super().__init__(init_cfg)
        if not isinstance(num_feats, list):
            num_feats = [num_feats] * 3
        self.h_embed = nn.Embedding(h, num_feats[0])
        self.w_embed = nn.Embedding(w, num_feats[1])
        self.z_embed = nn.Embedding(z, num_feats[2])
        self.num_feats = num_feats
        self.h, self.w, self.z = h, w, z

    def forward(self, bs, device, ignore_axis='z'):
        if ignore_axis == 'h':
            h_embed = torch.zeros(
                1, 1, self.num_feats[0],
                device=device).repeat(self.w, self.z, 1)  # w, z, d
            w_embed = self.w_embed(torch.arange(self.w, device=device))
            w_embed = w_embed.reshape(self.w, 1, -1).repeat(1, self.z, 1)
            z_embed = self.z_embed(torch.arange(self.z, device=device))
            z_embed = z_embed.reshape(1, self.z, -1).repeat(self.w, 1, 1)
        elif ignore_axis == 'w':
            h_embed = self.h_embed(torch.arange(self.h, device=device))
            h_embed = h_embed.reshape(1, self.h, -1).repeat(self.z, 1, 1)
            w_embed = torch.zeros(
                1, 1, self.num_feats[1],
                device=device).repeat(self.z, self.h, 1)
            z_embed = self.z_embed(torch.arange(self.z, device=device))
            z_embed = z_embed.reshape(self.z, 1, -1).repeat(1, self.h, 1)
        elif ignore_axis == 'z':
            h_embed = self.h_embed(torch.arange(self.h, device=device))
            h_embed = h_embed.reshape(self.h, 1, -1).repeat(1, self.w, 1)
            w_embed = self.w_embed(torch.arange(self.w, device=device))
            w_embed = w_embed.reshape(1, self.w, -1).repeat(self.h, 1, 1)
            z_embed = torch.zeros(
                1, 1, self.num_feats[2],
                device=device).repeat(self.h, self.w, 1)

        pos = torch.cat((h_embed, w_embed, z_embed),
                        dim=-1).flatten(0, 1).unsqueeze(0).repeat(bs, 1, 1)
        return pos
