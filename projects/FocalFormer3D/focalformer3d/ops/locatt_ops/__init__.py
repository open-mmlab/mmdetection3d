"""CUDA code credit: https://github.com/zzd1992/Image-Local-Attention."""

import os

import torch
from torch.utils import cpp_extension

cwd = os.path.dirname(os.path.realpath(__file__))

sources = []

assert torch.cuda.is_available(), 'local attention needs cuda!'
if torch.cuda.is_available():
    sources.append(os.path.join(cwd, 'similar.cu'))
    sources.append(os.path.join(cwd, 'weighting.cu'))
    sources.append(os.path.join(cwd, 'localAttention.cpp'))
extra_cuda_cflags = [
    '-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__'
]
localattention = cpp_extension.load(
    'localattention',
    sources=sources,
    build_directory=cwd,
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=False)
