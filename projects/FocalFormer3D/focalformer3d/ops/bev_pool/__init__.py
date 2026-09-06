import os

import torch
from torch.utils import cpp_extension

cwd = os.path.dirname(os.path.realpath(__file__))

sources = []

if torch.cuda.is_available():
    sources.append(os.path.join(cwd, 'src', 'bev_pool.cpp'))
    sources.append(os.path.join(cwd, 'src', 'bev_pool_cuda.cu'))

extra_cuda_cflags = [
    '-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__'
]

bev_pool_ext = cpp_extension.load(
    'bev_pool_ext',
    sources=sources,
    build_directory=cwd,
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=False)
