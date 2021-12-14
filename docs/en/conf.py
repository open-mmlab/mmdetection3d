# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import pytorch_sphinx_theme
import subprocess
import sys
from m2r import MdInclude
from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'MMDetection3D'
copyright = '2020-2023, OpenMMLab'
author = 'MMDetection3D Authors'

version_file = '../mmdet3d/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


# The full version, including alpha/beta/rc tags
release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'recommonmark',
    'sphinx_markdown_tables',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
]

autodoc_mock_imports = [
    'matplotlib', 'nuscenes', 'PIL', 'pycocotools', 'pyquaternion',
    'terminaltables', 'mmdet3d.version', 'mmdet3d.ops'
]
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    # 'logo_url': 'https://mmocr.readthedocs.io/en/latest/',
    'menu': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-mmlab/mmdetection3d'
        },
        {
            'name':
            'Projects',
            'children': [{
                'name':
                'MMCV',
                'url':
                'https://mmcv.readthedocs.io/en/latest/',
                'description':
                'Foundational library for computer vision'
            }, {
                'name':
                'MMDetection',
                'url':
                'https://mmdetection.readthedocs.io/en/latest/',
                'description':
                'Object detection toolbox and benchmark'
            }, {
                'name':
                'MMAction2',
                'url':
                'https://mmaction2.readthedocs.io/en/latest/',
                'description':
                'Action understanding toolbox and benchmark'
            }, {
                'name':
                'MMClassification',
                'url':
                'https://mmclassification.readthedocs.io/en/latest/',
                'description':
                'Image classification toolbox and benchmark'
            }, {
                'name':
                'MMSegmentation',
                'url':
                'https://mmsegmentation.readthedocs.io/en/latest/',
                'description':
                'Semantic segmentation toolbox and benchmark'
            }, {
                'name': 'MMEditing',
                'url': 'https://mmediting.readthedocs.io/en/latest/',
                'description': 'Image and video editing toolbox'
            }, {
                'name':
                'MMOCR',
                'url':
                'https://mmocr.readthedocs.io/en/latest/',
                'description':
                'Text detection, recognition and understanding toolbox'
            }, {
                'name': 'MMPose',
                'url': 'https://mmpose.readthedocs.io/en/latest/',
                'description': 'Pose estimation toolbox and benchmark'
            }, {
                'name':
                'MMTracking',
                'url':
                'https://mmtracking.readthedocs.io/en/latest/',
                'description':
                'Video perception toolbox and benchmark'
            }, {
                'name': 'MMGeneration',
                'url': 'https://mmgeneration.readthedocs.io/en/latest/',
                'description': 'Generative model toolbox'
            }, {
                'name': 'MMFlow',
                'url': 'https://mmflow.readthedocs.io/en/latest/',
                'description': 'Optical flow toolbox and benchmark'
            }, {
                'name':
                'MMFewShot',
                'url':
                'https://mmfewshot.readthedocs.io/en/latest/',
                'description':
                'FewShot learning toolbox and benchmark'
            }, {
                'name':
                'MMHuman3D',
                'url':
                'https://mmhuman3d.readthedocs.io/en/latest/',
                'description':
                '3D human parametric model toolbox and benchmark.'
            }]
        },
        {
            'name':
            'OpenMMLab',
            'children': [
                {
                    'name': 'Homepage',
                    'url': 'https://openmmlab.com/'
                },
                {
                    'name': 'GitHub',
                    'url': 'https://github.com/open-mmlab/'
                },
                {
                    'name': 'Twitter',
                    'url': 'https://twitter.com/OpenMMLab'
                },
                {
                    'name': 'Zhihu',
                    'url': 'https://zhihu.com/people/openmmlab'
                },
            ]
        },
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']

latex_documents = [
    (master_doc, 'mmcv.tex', 'mmcv Documentation', 'MMCV Contributors',
     'manual'),
]

# -- Extension configuration -------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True


def builder_inited_handler(app):
    subprocess.run(['./stat.py'])


def setup(app):
    app.connect('builder-inited', builder_inited_handler)
    app.add_config_value('no_underscore_emphasis', False, 'env')
    app.add_config_value('m2r_parse_relative_links', False, 'env')
    app.add_config_value('m2r_anonymous_references', False, 'env')
    app.add_config_value('m2r_disable_inline_math', False, 'env')
    app.add_directive('mdinclude', MdInclude)
    app.add_config_value('recommonmark_config', {
        'auto_toc_tree_section': 'Contents',
        'enable_eval_rst': True,
    }, True)
    app.add_transform(AutoStructify)
