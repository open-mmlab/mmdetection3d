import numpy as np
import torch


class Residual3DBoxCoder(object):

    def __init__(self, code_size=7, mean=None, std=None):
        super().__init__()
        self.code_size = code_size
        self.mean = mean
        self.std = std

    @staticmethod
    def encode_np(boxes, anchors):
        """
        :param boxes: (N, 7) x, y, z, w, l, h, r
        :param anchors: (N, 7)
        :return:
        """
        # need to convert boxes to z-center format
        xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
        xg, yg, zg, wg, lg, hg, rg = np.split(boxes, 7, axis=-1)
        zg = zg + hg / 2
        za = za + ha / 2
        diagonal = np.sqrt(la**2 + wa**2)  # 4.3
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha  # 1.6
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)
        rt = rg - ra
        return np.concatenate([xt, yt, zt, wt, lt, ht, rt], axis=-1)

    @staticmethod
    def decode_np(box_encodings, anchors):
        """
        :param box_encodings: (N, 7) x, y, z, w, l, h, r
        :param anchors: (N, 7)
        :return:
        """
        # need to convert box_encodings to z-bottom format
        xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
        xt, yt, zt, wt, lt, ht, rt = np.split(box_encodings, 7, axis=-1)

        za = za + ha / 2
        diagonal = np.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        return np.concatenate([xg, yg, zg, wg, lg, hg, rg], axis=-1)

    @staticmethod
    def encode_torch(anchors, boxes, means, stds):
        """
        :param boxes: (N, 7+n) x, y, z, w, l, h, r, velo*
        :param anchors: (N, 7+n)
        :return:
        """
        box_ndim = anchors.shape[-1]
        cas, cgs, cts = [], [], []
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
            xg, yg, zg, wg, lg, hg, rg, *cgs = torch.split(boxes, 1, dim=-1)
            cts = [g - a for g, a in zip(cgs, cas)]
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
        za = za + ha / 2
        zg = zg + hg / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt, *cts], dim=-1)

    @staticmethod
    def decode_torch(anchors, box_encodings, means, stds):
        """
        :param box_encodings: (N, 7 + n) x, y, z, w, l, h, r
        :param anchors: (N, 7)
        :return:
        """
        cas, cts = [], []
        box_ndim = anchors.shape[-1]
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(
                box_encodings, 1, dim=-1)
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)
