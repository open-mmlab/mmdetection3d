from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    CppExtension(
        name='points_op_cpu',
        sources = ['src/points_op.cpp'],
        extra_compile_args=['-g'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
    ),
]

setup(
    name='cpplib',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    })

