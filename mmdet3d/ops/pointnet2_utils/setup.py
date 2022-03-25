from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointnet2_utils',
    ext_modules=[
        CUDAExtension('pointnet2_cuda', [
            'src/pointnet2_api.cpp',
            'src/interpolate.cpp', 
            'src/interpolate_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
