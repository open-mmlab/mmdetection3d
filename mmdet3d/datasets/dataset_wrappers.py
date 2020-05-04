import math
from collections import defaultdict

import numpy as np

from mmdet.datasets import DATASETS


# Modified from https://github.com/facebookresearch/detectron2/blob/41d475b75a230221e21d9cac5d69655e3415e3a4/detectron2/data/samplers/distributed_sampler.py#L57 # noqa
@DATASETS.register_module()
class RepeatFactorDataset(object):
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. In each
    epoch, an image may appear multiple times based on its "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined as the fraction of images in the training set (without repeats)
    in which category c appears.
    This wrapper will finally be merged into LVIS dataset.

    See https://arxiv.org/abs/1908.03195 (>= v2) Appendix B.2.
    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        repeat_thr (float): frequency threshold below which data is repeated.
    """

    def __init__(self, dataset, repeat_thr):
        self.dataset = dataset
        self.repeat_thr = repeat_thr
        self.CLASSES = dataset.CLASSES

        repeat_factors = self._get_repeat_factors(dataset, repeat_thr)
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        # 1. For each category c, compute the fraction # of images
        # that contain it: f(c)
        category_freq = defaultdict(int)
        for idx, img_info in enumerate(dataset.data_infos):
            if 'category_ids' in img_info:
                cat_ids = set(img_info['category_ids'])
            elif 'gt_names' in img_info:
                cat_ids = set([
                    gt for gt in img_info['gt_names']
                    if gt in dataset.class_names
                ])
            else:
                labels = dataset.get_ann_info(idx)['labels']
                cat_ids = set([label for label in labels])
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx, img_info in enumerate(dataset.data_infos):
            if 'category_ids' in img_info:
                cat_ids = set(img_info['category_ids'])
            elif 'gt_names' in img_info:
                cat_ids = set([
                    gt for gt in img_info['gt_names']
                    if gt in dataset.class_names
                ])
            else:
                labels = dataset.get_ann_info(idx)['labels']
                cat_ids = set([label for label in labels])

            if len(cat_ids) == 0:
                repeat_factor = 1
            else:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)
        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        return len(self.repeat_indices)
