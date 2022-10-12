# FAQ

We list some potential troubles encountered by users and developers, along with their corresponding solutions. Feel free to enrich the list if you find any frequent issues and contribute your solutions to solve them. If you have any trouble with environment configuration, model training, etc, please create an issue using the [provided templates](https://github.com/open-mmlab/mmdetection3d/blob/master/.github/ISSUE_TEMPLATE/error-report.md/) and fill in all required information in the template.

## MMCV/MMDet/MMDet3D Installation

- Compatibility issue between MMCV, MMDetection, MMSegmentation and MMDection3D; "ConvWS is already registered in conv layer"; "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, \<=xxx."

  The required versions of MMCV, MMDetection and MMSegmentation for different versions of MMDetection3D are as below. Please install the correct version of MMCV, MMDetection and MMSegmentation to avoid installation issues.

  | MMDetection3D version |   MMDetection version    | MMSegmentation version  |        MMCV version         |
  | :-------------------: | :----------------------: | :---------------------: | :-------------------------: |
  |        master         | mmdet>=2.24.0, \<=3.0.0  | mmseg>=0.20.0, \<=1.0.0 | mmcv-full>=1.5.2, \<=1.7.0  |
  |       v1.0.0rc5       | mmdet>=2.24.0, \<=3.0.0  | mmseg>=0.20.0, \<=1.0.0 | mmcv-full>=1.5.2, \<=1.7.0  |
  |       v1.0.0rc4       | mmdet>=2.24.0, \<=3.0.0  | mmseg>=0.20.0, \<=1.0.0 | mmcv-full>=1.5.2, \<=1.7.0  |
  |       v1.0.0rc3       | mmdet>=2.24.0, \<=3.0.0  | mmseg>=0.20.0, \<=1.0.0 | mmcv-full>=1.4.8, \<=1.6.0  |
  |       v1.0.0rc2       | mmdet>=2.24.0, \<=3.0.0  | mmseg>=0.20.0, \<=1.0.0 | mmcv-full>=1.4.8, \<=1.6.0  |
  |       v1.0.0rc1       | mmdet>=2.19.0, \<=3.0.0  | mmseg>=0.20.0, \<=1.0.0 | mmcv-full>=1.4.8, \<=1.5.0  |
  |       v1.0.0rc0       | mmdet>=2.19.0, \<=3.0.0  | mmseg>=0.20.0, \<=1.0.0 | mmcv-full>=1.3.17, \<=1.5.0 |
  |        0.18.1         | mmdet>=2.19.0, \<=3.0.0  | mmseg>=0.20.0, \<=1.0.0 | mmcv-full>=1.3.17, \<=1.5.0 |
  |        0.18.0         | mmdet>=2.19.0, \<=3.0.0  | mmseg>=0.20.0, \<=1.0.0 | mmcv-full>=1.3.17, \<=1.5.0 |
  |        0.17.3         | mmdet>=2.14.0, \<=3.0.0  | mmseg>=0.14.1, \<=1.0.0 | mmcv-full>=1.3.8, \<=1.4.0  |
  |        0.17.2         | mmdet>=2.14.0, \<=3.0.0  | mmseg>=0.14.1, \<=1.0.0 | mmcv-full>=1.3.8, \<=1.4.0  |
  |        0.17.1         | mmdet>=2.14.0, \<=3.0.0  | mmseg>=0.14.1, \<=1.0.0 | mmcv-full>=1.3.8, \<=1.4.0  |
  |        0.17.0         | mmdet>=2.14.0, \<=3.0.0  | mmseg>=0.14.1, \<=1.0.0 | mmcv-full>=1.3.8, \<=1.4.0  |
  |        0.16.0         | mmdet>=2.14.0, \<=3.0.0  | mmseg>=0.14.1, \<=1.0.0 | mmcv-full>=1.3.8, \<=1.4.0  |
  |        0.15.0         | mmdet>=2.14.0, \<=3.0.0  | mmseg>=0.14.1, \<=1.0.0 | mmcv-full>=1.3.8, \<=1.4.0  |
  |        0.14.0         | mmdet>=2.10.0, \<=2.11.0 |      mmseg==0.14.0      | mmcv-full>=1.3.1, \<=1.4.0  |
  |        0.13.0         | mmdet>=2.10.0, \<=2.11.0 |      Not required       | mmcv-full>=1.2.4, \<=1.4.0  |
  |        0.12.0         | mmdet>=2.5.0, \<=2.11.0  |      Not required       | mmcv-full>=1.2.4, \<=1.4.0  |
  |        0.11.0         | mmdet>=2.5.0, \<=2.11.0  |      Not required       | mmcv-full>=1.2.4, \<=1.3.0  |
  |        0.10.0         | mmdet>=2.5.0, \<=2.11.0  |      Not required       | mmcv-full>=1.2.4, \<=1.3.0  |
  |         0.9.0         | mmdet>=2.5.0, \<=2.11.0  |      Not required       | mmcv-full>=1.2.4, \<=1.3.0  |
  |         0.8.0         | mmdet>=2.5.0, \<=2.11.0  |      Not required       | mmcv-full>=1.1.5, \<=1.3.0  |
  |         0.7.0         | mmdet>=2.5.0, \<=2.11.0  |      Not required       | mmcv-full>=1.1.5, \<=1.3.0  |
  |         0.6.0         | mmdet>=2.4.0, \<=2.11.0  |      Not required       | mmcv-full>=1.1.3, \<=1.2.0  |
  |         0.5.0         |          2.3.0           |      Not required       |      mmcv-full==1.0.5       |

- If you faced the error shown below when importing open3d:

  `OSError: /lib/x86_64-linux-gnu/libm.so.6: version 'GLIBC_2.27' not found`

  please downgrade open3d to 0.9.0.0, because the latest open3d needs the support of file 'GLIBC_2.27', which only exists in Ubuntu 18.04, not in Ubuntu 16.04.

- If you faced the error when importing pycocotools, this is because nuscenes-devkit installs pycocotools but mmdet relies on mmpycocotools. The current workaround is as below. We will migrate to use pycocotools in the future.

  ```shell
  pip uninstall pycocotools mmpycocotools
  pip install mmpycocotools
  ```

  **NOTE**: We have migrated to use pycocotools in mmdet3d >= 0.13.0.

- If you face the error shown below when importing pycocotools:

  `ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject`

  please downgrade pycocotools to 2.0.1 because of the incompatibility between the newest pycocotools and numpy \< 1.20.0. Or you can compile and install the latest pycocotools from source as below:

  `pip install -e "git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI"`

  or

  `pip install -e "git+https://github.com/ppwwyyxx/cocoapi#egg=pycocotools&subdirectory=PythonAPI"`

## How to annotate point cloud?

MMDetection3D does not support point cloud annotation. Some open-source annotation tool are offered for reference:

- [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS)
- [LATTE](https://github.com/bernwang/latte)

Besides, we improved [LATTE](https://github.com/bernwang/latte) for better use. More details can be found [here](https://arxiv.org/abs/2011.10174).
