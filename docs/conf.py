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
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'MMDetection3D'
copyright = '2020-2023, OpenMMLab'
author = 'MMDetection3D Authors'

# The full version, including alpha/beta/rc tags
with open('../mmdet3d/VERSION', 'r') as f:
    release = f.read().strip()

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
]

autodoc_mock_imports = [
    'cv2', 'matplotlib', 'nuscenes', 'PIL', 'pycocotools', 'pyquaternion',
    'terminaltables', 'mmcv', 'mmdet', 'mmdet3d.version',
    'mmdet3d.ops.ball_query', 'mmdet3d.ops.furthest_point_sample',
    'mmdet3d.ops.gather_points', 'mmdet3d.ops.group_points',
    'mmdet3d.ops.interpolate', 'mmdet3d.ops.roiaware_pool3d',
    'mmdet3d.ops.spconv', 'mmdet3d.ops.voxel.voxel_layer', 'mmdet3d.ops.iou3d'
]

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
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
