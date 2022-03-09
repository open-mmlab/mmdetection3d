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

  **NOTE**: We have migrated to use pycocotools in mmdet3d >= 0.13.0.

- If you face the error shown below when importing pycocotools:

  ``ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject``

  please downgrade pycocotools to 2.0.1 because of the incompatibility between the newest pycocotools and numpy < 1.20.0. Or you can compile and install the latest pycocotools from source as below:

  ``pip install -e "git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI"``

  or

  ``pip install -e "git+https://github.com/ppwwyyxx/cocoapi#egg=pycocotools&subdirectory=PythonAPI"``

## How to annotate point cloud?

MMDetection3D does not support point cloud annotation. Some open-source annotation tool are offered for reference:

- [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS)
- [LATTE](https://github.com/bernwang/latte)

Besides, we improved [LATTE](https://github.com/bernwang/latte) for better use. More details can be found [here](https://arxiv.org/abs/2011.10174).
