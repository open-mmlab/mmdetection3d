# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DSegmentor
from .encoder_decoder import EncoderDecoder3D
from .minkunet import MinkUNet

__all__ = ['Base3DSegmentor', 'EncoderDecoder3D', 'MinkUNet']
