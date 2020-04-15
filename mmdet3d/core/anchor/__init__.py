from .anchor_generator import (AlignedAnchorGeneratorRange, AnchorGenerator,
                               AnchorGeneratorRange)

__all__ = [
    'AnchorGenerator', 'AlignedAnchorGeneratorRange', 'AnchorGeneratorRange',
    'build_anchor_generator'
]


def build_anchor_generator(cfg, **kwargs):
    from . import anchor_generator
    import mmcv
    if isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(
            cfg, anchor_generator, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))
