# Copyright (c) OpenMMLab. All rights reserved.

third_part_libs = [
    'conda install openblas-devel -c anaconda',
    "pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option='--blas_include_dirs=/opt/conda/include' --install-option='--blas=openblas'"  # noqa
]
default_floating_range = 0.5
model_floating_ranges = {
    'configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py': # noqa
    0.7
}
