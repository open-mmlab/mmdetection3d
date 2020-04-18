from .anchor_3d_generator import (AlignedAnchorGeneratorRange,
                                  AnchorGeneratorRange)

__all__ = [
    'AlignedAnchorGeneratorRange', 'AnchorGeneratorRange',
    'build_anchor_generator'
]


def build_anchor_generator(cfg, **kwargs):
    from . import anchor_3d_generator
    import mmcv
    if isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(
            cfg, anchor_3d_generator, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))
