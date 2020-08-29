import torch

from mmdet.core.bbox.builder import BBOX_SAMPLERS
from . import RandomSampler, SamplingResult


@BBOX_SAMPLERS.register_module()
class IoUNegPiecewiseSampler(RandomSampler):
    """IoU Piece-wise Sampling.

    Sampling negtive proposals according to a list of IoU thresholds.
    The negtive proposals are divided into several pieces according
    to `neg_iou_piece_thrs`. And the ratio of each piece is indicated
    by `neg_piece_fractions`.

    Args:
        num (int): Number of proposals.
        pos_fraction (float): The fraction of positive proposals.
        neg_piece_fractions (list): A list contains fractions that indicates
            the ratio of each piece of total negtive samplers.
        neg_iou_piece_thrs (list): A list contains IoU thresholds that
            indicate the upper bound of this piece.
        neg_pos_ub (float): The total ratio to limit the upper bound
            number of negtive samples.
        add_gt_as_proposals (bool): Whether to add gt as proposals.
    """

    def __init__(self,
                 num,
                 pos_fraction=None,
                 neg_piece_fractions=None,
                 neg_iou_piece_thrs=None,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=False,
                 return_iou=False):
        super(IoUNegPiecewiseSampler,
              self).__init__(num, pos_fraction, neg_pos_ub,
                             add_gt_as_proposals)
        assert isinstance(neg_piece_fractions, list)
        assert len(neg_piece_fractions) == len(neg_iou_piece_thrs)
        self.neg_piece_fractions = neg_piece_fractions
        self.neg_iou_thr = neg_iou_piece_thrs
        self.return_iou = return_iou
        self.neg_piece_num = len(self.neg_piece_fractions)

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            neg_inds_choice = neg_inds.new_zeros([0])
            extend_num = 0
            max_overlaps = assign_result.max_overlaps[neg_inds]

            for piece_inds in range(self.neg_piece_num):
                if piece_inds == self.neg_piece_num - 1:  # for the last piece
                    piece_expected_num = num_expected - len(neg_inds_choice)
                    min_iou_thr = 0
                else:
                    # if the numbers of negative samplers in previous
                    # pieces are less than the expected number, extend
                    # the same number in the current piece.
                    piece_expected_num = int(
                        num_expected *
                        self.neg_piece_fractions[piece_inds]) + extend_num
                    min_iou_thr = self.neg_iou_thr[piece_inds + 1]
                max_iou_thr = self.neg_iou_thr[piece_inds]
                piece_neg_inds = torch.nonzero(
                    (max_overlaps >= min_iou_thr)
                    & (max_overlaps < max_iou_thr),
                    as_tuple=False).view(-1)

                if len(piece_neg_inds) < piece_expected_num:
                    neg_inds_choice = torch.cat(
                        [neg_inds_choice, neg_inds[piece_neg_inds]], dim=0)
                    extend_num += piece_expected_num - len(piece_neg_inds)
                else:
                    piece_choice = self.random_choice(piece_neg_inds,
                                                      piece_expected_num)
                    neg_inds_choice = torch.cat(
                        [neg_inds_choice, neg_inds[piece_choice]], dim=0)
                    extend_num = 0
            return neg_inds_choice

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (torch.Tensor): Boxes to be sampled from.
            gt_bboxes (torch.Tensor): Ground truth bboxes.
            gt_labels (torch.Tensor, optional): Class labels of ground truth \
                bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.bool)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.bool)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        if self.return_iou:
            # PartA2 needs iou score to regression.
            sampling_result.iou = assign_result.max_overlaps[torch.cat(
                [pos_inds, neg_inds])]
            sampling_result.iou.detach_()

        return sampling_result
