from setuptools import find_packages, setup

import os
import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'mmdet3d/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    import sys

    # return short version for sdist
    if 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        return locals()['short_version']
    else:
        return locals()['__version__']


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        list[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    setup(
        name='mmdet3d',
        version=get_version(),
        description=("OpenMMLab's next-generation platform"
                     'for general 3D object detection.'),
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='OpenMMLab',
        author_email='zwwdev@gmail.com',
        keywords='computer vision, 3D object detection',
        url='https://github.com/open-mmlab/mmdetection3d',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        package_data={'mmdet3d.ops': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=[
            make_cuda_ext(
                name='sparse_conv_ext',
                module='mmdet3d.ops.spconv',
                extra_include_path=[
                    # PyTorch 1.5 uses ninjia, which requires absolute path
                    # of included files, relative path will cause failure.
                    os.path.abspath(
                        os.path.join(*'mmdet3d.ops.spconv'.split('.'),
                                     'include/'))
                ],
                sources=[
                    'src/all.cc',
                    'src/reordering.cc',
                    'src/reordering_cuda.cu',
                    'src/indice.cc',
                    'src/indice_cuda.cu',
                    'src/maxpool.cc',
                    'src/maxpool_cuda.cu',
                ],
                extra_args=['-w', '-std=c++14']),
            make_cuda_ext(
                name='iou3d_cuda',
                module='mmdet3d.ops.iou3d',
                sources=[
                    'src/iou3d.cpp',
                    'src/iou3d_kernel.cu',
                ]),
            make_cuda_ext(
                name='voxel_layer',
                module='mmdet3d.ops.voxel',
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cpu.cpp',
                    'src/scatter_points_cuda.cu',
                    'src/voxelization_cpu.cpp',
                    'src/voxelization_cuda.cu',
                ]),
            make_cuda_ext(
                name='roiaware_pool3d_ext',
                module='mmdet3d.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/points_in_boxes_cpu.cpp',
                ],
                sources_cuda=[
                    'src/roiaware_pool3d_kernel.cu',
                    'src/points_in_boxes_cuda.cu',
                ]),
            make_cuda_ext(
                name='ball_query_ext',
                module='mmdet3d.ops.ball_query',
                sources=['src/ball_query.cpp'],
                sources_cuda=['src/ball_query_cuda.cu']),
            make_cuda_ext(
                name='group_points_ext',
                module='mmdet3d.ops.group_points',
                sources=['src/group_points.cpp'],
                sources_cuda=['src/group_points_cuda.cu']),
            make_cuda_ext(
                name='interpolate_ext',
                module='mmdet3d.ops.interpolate',
                sources=['src/interpolate.cpp'],
                sources_cuda=[
                    'src/three_interpolate_cuda.cu', 'src/three_nn_cuda.cu'
                ]),
            make_cuda_ext(
                name='furthest_point_sample_ext',
                module='mmdet3d.ops.furthest_point_sample',
                sources=['src/furthest_point_sample.cpp'],
                sources_cuda=['src/furthest_point_sample_cuda.cu']),
            make_cuda_ext(
                name='gather_points_ext',
                module='mmdet3d.ops.gather_points',
                sources=['src/gather_points.cpp'],
                sources_cuda=['src/gather_points_cuda.cu'])
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
