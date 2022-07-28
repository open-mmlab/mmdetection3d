# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DSegmentor
from .encoder_decoder import EncoderDecoder3D
from .sparse_encoder_decoder import SparseEncoderDecoder3D

__all__ = ['Base3DSegmentor', 'EncoderDecoder3D', 'SparseEncoderDecoder3D']
