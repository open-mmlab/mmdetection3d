import numpy as np

from .builder import DATASETS


@DATASETS.register_module()
class ClassSampledDataset(object):
    """A wrapper of class sampled dataset with ann_file path.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
        ann_file (str): Path of annotation file.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.repeat_indices = self._get_sample_indices()
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, 'flag'):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.repeat_indices],
                dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        data = self.dataset.data_infos
        class_sample_idx = {name: [] for name in self.CLASSES}
        for idx, info in enumerate(data):
            if self.dataset.use_valid_flag:
                mask = info['valid_flag']
                gt_names = set(info['gt_names'][mask])
            else:
                gt_names = set(info['gt_names'])
            for name in gt_names:
                if name in self.CLASSES:
                    class_sample_idx[name].append(idx)
        duplicated_samples = sum([len(v) for _, v in class_sample_idx.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idx.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_infos, ratio in zip(list(class_sample_idx.values()), ratios):
            sample_indices += np.random.choice(cls_infos,
                                               int(len(cls_infos) *
                                                   ratio)).tolist()
        return sample_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        ori_idx = self.repeat_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)
