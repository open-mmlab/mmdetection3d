from mmengine.registry import DATASETS
from mmengine.dataset import ConcatDataset

@DATASETS.register_module()
class SemiDataset(ConcatDataset):
    """Semi-supervised dataset to handle labeled and unlabeled data.

    This dataset inherits from `ConcatDataset` and adds specific handling for
    labeled (source) and unlabeled (target) data.

    Args:
        datasets (list[dict]): A list of datasets to be concatenated.
            The first dataset is assumed to be the labeled dataset (source).
            The second dataset is assumed to be the unlabeled dataset (target).
    """
    def __init__(self, datasets, **kwargs):
        super().__init__(datasets, **kwargs)
        if len(self.datasets) != 2:
            raise ValueError("SemiDataset requires exactly two datasets: "
                             "labeled (source) and unlabeled (target).")
        self.labeled_dataset = self.datasets[0]
        self.unlabeled_dataset = self.datasets[1]

    def __getitem__(self, idx):
        """Get a sample, distinguishing between labeled and unlabeled data."""
        if idx < len(self.labeled_dataset):
            # Sample from the labeled dataset
            sample = self.labeled_dataset[idx]
            sample['is_labeled'] = True
        else:
            # Sample from the unlabeled dataset
            unlabeled_idx = idx - len(self.labeled_dataset)
            sample = self.unlabeled_dataset[unlabeled_idx]
            # Add placeholder for pseudo-labels
            # sample['gt_bboxes_3d'] = None
            # sample['gt_labels_3d'] = None
            sample['is_labeled'] = False
        return sample

    def get_dataset_source(self, idx):
        """Determine the dataset source for a given index."""
        if idx < len(self.labeled_dataset):
            return 'labeled'
        else:
            return 'unlabeled'
