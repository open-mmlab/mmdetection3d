# FAQ

We list some potential troubles encountered by users and developers, along with their corresponding solutions. Feel free to enrich the list if you find any frequent issues and contribute your solutions to solve them. If you have any trouble with environment configuration, model training, etc, please create an issue using the [provided templates](https://github.com/open-mmlab/mmdetection3d/blob/master/.github/ISSUE_TEMPLATE/error-report.md/) and fill in all required information in the template.

## MMCV/MMDet/MMDet3D Installation

- If you faced the error shown below when importing open3d:

  ``OSError: /lib/x86_64-linux-gnu/libm.so.6: version 'GLIBC_2.27' not found``

  please downgrade open3d to 0.9.0.0, because the latest open3d needs the support of file 'GLIBC_2.27', which only exists in Ubuntu 18.04, not in Ubuntu 16.04.

- If you faced the error when importing pycocotools, this is because nuscenes-devkit installs pycocotools but mmdet relies on mmpycocotools. The current workaround is as below. We will migrate to use pycocotools in the future.

  ```shell
  pip uninstall pycocotools mmpycocotools
  pip install mmpycocotools
  ```

- If you face the error shown below, and your environment contains numba == 0.48.0 with numpy >= 1.20.0

  ``TypeError: expected dtype object, got 'numpy.dtype[bool_]'``

  please downgrade numpy to < 1.20.0 or install numba == 0.48 from source, because in numpy==1.20.0, `np.dtype` produces subclass due to API change.
