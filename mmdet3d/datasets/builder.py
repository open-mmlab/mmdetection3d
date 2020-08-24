import copy
import platform
from mmcv.utils import build_from_cfg

from mmdet.datasets import DATASETS

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def _concat_dataset(cfg, default_args=None):
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import ClassSampledDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'ClassSampledDataset':
        dataset = ClassSampledDataset(
            build_dataset(cfg['dataset'], default_args),
            cfg['dataset']['ann_file'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
