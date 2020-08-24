import mmcv
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

    def __init__(self, dataset, ann_file):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag
        self._ori_len = len(self.dataset)
        self.dataset.data_infos = self.load_annotations(ann_file)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        data = mmcv.load(ann_file)
        _cls_infos = {name: [] for name in self.CLASSES}
        for info in data['infos']:
            if self.dataset.use_valid_flag:
                mask = info['valid_flag']
                gt_names = set(info['gt_names'][mask])
            else:
                gt_names = set(info['gt_names'])
            for name in gt_names:
                if name in self.CLASSES:
                    _cls_infos[name].append(info)
        duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
        _cls_dist = {
            k: len(v) / duplicated_samples
            for k, v in _cls_infos.items()
        }

        data_infos = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in _cls_dist.values()]
        for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
            data_infos += np.random.choice(cls_infos,
                                           int(len(cls_infos) *
                                               ratio)).tolist()

        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        return self.dataset[idx]
