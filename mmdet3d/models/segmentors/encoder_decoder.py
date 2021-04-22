import numpy as np
import torch
from torch.nn import functional as F

from mmseg.models import SEGMENTORS, EncoderDecoder, build_head
from ..builder import build_backbone, build_neck
from .base import Base3DSegmentor


@SEGMENTORS.register_module()
class EncoderDecoder3D(Base3DSegmentor, EncoderDecoder):
    """3D Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder3D, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = build_head(decode_head)
        self.num_classes = self.decode_head.num_classes

    def encode_decode(self, points, img_metas):
        """Encode points with backbone and decode into a semantic segmentation
        map of the same size as input.

        Args:
            points (torch.Tensor): Input points of shape [B, N, 3+C].
            img_metas (list[dict]): Meta information of each sample.
        """
        x = self.extract_feat(points)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    @staticmethod
    def _input_generation(coords,
                          patch_center,
                          coord_max,
                          feats,
                          use_normalized_coord=False):
        """Generating model input.

        Generate input by subtracting patch center and adding additional \
            features. Currently support colors and normalized xyz as features.

        Args:
            coords (torch.Tensor): Sampled 3D point coordinate of shape [S, 3].
            patch_center (torch.Tensor): Center coordinate of the patch.
            coord_max (torch.Tensor): Max coordinate of all 3D points.
            feats (torch.Tensor): Features of sampled points of shape [S, C].
            use_normalized_coord (bool, optional): Whether to use normalized \
                xyz as additional features. Defaults to False.

        Returns:
            torch.Tensor: The generated input data of shape [S, 3+C'].
        """
        # subtract patch center, the z dimension is not centered
        centered_coords = coords.clone()
        centered_coords[:, 0] -= patch_center[0]
        centered_coords[:, 1] -= patch_center[1]

        # normalized coordinates as extra features
        if use_normalized_coord:
            normalized_coord = coords / coord_max
            feats = torch.cat([feats, normalized_coord], dim=1)

        points = torch.cat([centered_coords, feats], dim=1)

        return points

    def _sliding_patch_generation(self,
                                  points,
                                  num_points,
                                  block_size,
                                  sample_rate=0.5,
                                  use_normalized_coord=False):
        """Sampling points in a sliding window fashion.

        First sample patches to cover all the input points.
        Then sample points in each patch to batch points of a certain number.

        Args:
            points (torch.Tensor): Input points of shape [N, 3+C].
            num_points (int): Number of points to be sampled in each patch.
            block_size (float, optional): Size of a patch to sample.
            sample_rate (float, optional): Stride used in sliding patch.
                Defaults to 0.5.
            use_normalized_coord (bool, optional): Whether to use normalized \
                xyz as additional features. Defaults to False.

        Returns:
            np.ndarray | np.ndarray:

                - patch_points (torch.Tensor): Points of different patches of \
                    shape [K, N, 3+C].
                - patch_idxs (torch.Tensor): Index of each point in \
                    `patch_points`, of shape [K, N].
        """
        device = points.device
        coords = points[:, :3]
        feats = points[:, 3:]

        coord_max = coords.max(0)[0]
        coord_min = coords.min(0)[0]
        stride = block_size * sample_rate
        num_grid_x = int(
            torch.ceil((coord_max[0] - coord_min[0] - block_size) /
                       stride).item() + 1)
        num_grid_y = int(
            torch.ceil((coord_max[1] - coord_min[1] - block_size) /
                       stride).item() + 1)

        patch_points, patch_idxs = [], []
        for idx_y in range(num_grid_y):
            s_y = coord_min[1] + idx_y * stride
            e_y = torch.min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size
            for idx_x in range(num_grid_x):
                s_x = coord_min[0] + idx_x * stride
                e_x = torch.min(s_x + block_size, coord_max[0])
                s_x = e_x - block_size

                # extract points within this patch
                cur_min = torch.tensor([s_x, s_y, coord_min[2]]).to(device)
                cur_max = torch.tensor([e_x, e_y, coord_max[2]]).to(device)
                cur_choice = ((coords >= cur_min - 1e-3) &
                              (coords <= cur_max + 1e-3)).all(dim=1)

                if not cur_choice.any():  # no points in this patch
                    continue

                # sample points in this patch to multiple batches
                cur_center = cur_min + block_size / 2.0
                point_idxs = torch.nonzero(cur_choice, as_tuple=True)[0]
                num_batch = int(np.ceil(point_idxs.shape[0] / num_points))
                point_size = int(num_batch * num_points)
                replace = point_size > 2 * point_idxs.shape[0]

                # TODO: faster alternative of np.random.choice?
                repeat_idx = torch.ones_like(point_idxs).multinomial(
                    point_size - point_idxs.shape[0], replacement=replace)
                point_idxs_repeat = point_idxs[repeat_idx]
                choices = torch.cat([point_idxs, point_idxs_repeat], dim=0)
                choices = choices[torch.randperm(choices.shape[0])]

                # construct model input
                point_batches = self._input_generation(
                    coords[choices],
                    cur_center,
                    coord_max,
                    feats[choices],
                    use_normalized_coord=use_normalized_coord)

                patch_points.append(point_batches)
                patch_idxs.append(choices)

        patch_points = torch.cat(patch_points, dim=0)
        patch_idxs = torch.cat(patch_idxs, dim=0)

        # make sure all points are sampled at least once
        assert torch.unique(patch_idxs).shape[0] == points.shape[0]

        return patch_points, patch_idxs

    def slide_inference(self, point, img_meta):
        """Inference by sliding-window with overlap.

        Args:
            point (torch.Tensor): Input points of shape [N, 3+C].
            img_meta (dict): Meta information of input sample.

        Returns:
            Tensor: The output segmentation map of shape [num_classes, N].
        """
        num_points = self.test_cfg.num_points
        block_size = self.test_cfg.block_size
        sample_rate = self.test_cfg.sample_rate
        use_normalized_coord = self.test_cfg.use_normalized_coord
        batch_size = self.test_cfg.batch_size

        # patch_points is of shape [K, N, 3+C], patch_idxs is of shape [K, N]
        patch_points, patch_idxs = self._sliding_patch_generation(
            point, num_points, block_size, sample_rate, use_normalized_coord)
        preds = point.new_zeros((point.shape[0], self.num_classes))
        count_mat = point.new_zeros(point.shape[0])
        seg_logits = []  # save patch predictions

        for batch_idx in range(0, patch_points.shape[0], batch_size):
            batch_points = patch_points[batch_idx:batch_idx + batch_size]
            # batch_seg_logit is of shape [B, num_classes, N]
            batch_seg_logit = self.encode_decode(batch_points, img_meta)
            batch_seg_logit = batch_seg_logit.transpose(1, 2).contiguous()
            seg_logits.append(batch_seg_logit.view(-1, self.num_classes))

        # TODO: more efficient way to aggregate the predictions?
        seg_logits = torch.cat(seg_logits, dim=0)  # [K*N, num_classes]
        patch_idxs = patch_idxs.view(-1)  # [K*N]
        for i in range(patch_idxs.shape[0]):
            preds[patch_idxs[i]] += seg_logits[i]
            count_mat[patch_idxs[i]] += 1
        preds = preds / count_mat
        return preds.transpose(0, 1)  # to [num_classes, K*N]

    def whole_inference(self, points, img_metas):
        """Inference with full scene (one forward pass without sliding)."""
        seg_logit = self.encode_decode(points, img_metas)
        return seg_logit

    def inference(self, points, img_metas):
        """Inference with slide/whole style.

        Args:
            points (torch.Tensor): Input points of shape [B, N, 3+C].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            seg_logit = torch.stack([
                self.slide_inference(point, img_meta)
                for point, img_meta in zip(points, img_metas)
            ], 0)
        else:
            seg_logit = self.whole_inference(points, img_metas)
        output = F.softmax(seg_logit, dim=1)
        return output

    def simple_test(self, points, img_metas):
        """Simple test with single scene.

        Args:
            points (torch.Tensor): Input points of shape [1, N, 3+C].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            Tensor: The output segmentation map.
        """
        seg_logit = self.inference(points, img_metas)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, points, img_metas):
        """Test with augmentations.

        Args:
            points (torch.Tensor): Input points of shape [B, N, 3+C].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            Tensor: The output segmentation map.
        """
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(points[0], img_metas[0])
        for i in range(1, len(points)):
            cur_seg_logit = self.inference(points[i], img_metas[i])
            seg_logit += cur_seg_logit
        seg_logit /= len(points)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
