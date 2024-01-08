# 常见问题解答

我们列出了一些用户和开发者在开发过程中会遇到的常见问题以及对应的解决方案，如果您发现了任何频繁出现的问题，请随时扩充本列表，非常欢迎您提出的任何解决方案。如果您在环境配置、模型训练等工作中遇到任何的问题，请使用[问题模板](https://github.com/open-mmlab/mmdetection3d/blob/master/.github/ISSUE_TEMPLATE/error-report.md)来创建相应的 issue，并将所需的所有信息填入到问题模板中，我们会尽快解决您的问题。

## MMEngine/MMCV/MMDet/MMDet3D 安装

- 跟 MMEngine, MMCV, MMDetection 和 MMDetection3D 相关的编译问题; "ConvWS is already registered in conv layer"; "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, \<=xxx."

- MMDetection3D 需要的 MMEngine, MMCV 和 MMDetection 的版本列在了下面。请安装正确版本的 MMEngine、MMCV 和 MMDetection 以避免相关的安装问题。

  | MMDetection3D 版本 |      MMEngine 版本       |        MMCV 版本        |     MMDetection 版本     |
  | ------------------ | :----------------------: | :---------------------: | :----------------------: |
  | main               | mmengine>=0.8.0, \<1.0.0 | mmcv>=2.0.0rc4, \<2.2.0 | mmdet>=3.0.0rc5, \<3.4.0 |
  | v1.4.0             | mmengine>=0.8.0, \<1.0.0 | mmcv>=2.0.0rc4, \<2.2.0 | mmdet>=3.0.0rc5, \<3.4.0 |
  | v1.3.0             | mmengine>=0.8.0, \<1.0.0 | mmcv>=2.0.0rc4, \<2.2.0 | mmdet>=3.0.0rc5, \<3.3.0 |
  | v1.2.0             | mmengine>=0.8.0, \<1.0.0 | mmcv>=2.0.0rc4, \<2.1.0 |  mmdet>=3.0.0, \<3.2.0   |
  | v1.1.1             | mmengine>=0.7.1, \<1.0.0 | mmcv>=2.0.0rc4, \<2.1.0 |  mmdet>=3.0.0, \<3.1.0   |

  **注意**：如果你想安装 mmdet3d-v1.0.0rcx，可以在[此处](https://mmdetection3d.readthedocs.io/en/latest/faq.html#mmcv-mmdet-mmdet3d-installation)找到 MMDetection，MMSegmentation 和 MMCV 的兼容版本。请选择正确版本的 MMCV、MMDetection 和 MMSegmentation 以避免安装问题。

- 如果您在 `import open3d` 时遇到下面的问题：

  `OSError: /lib/x86_64-linux-gnu/libm.so.6: version 'GLIBC_2.27' not found`

  请将 open3d 的版本降级至 0.9.0.0，因为最新版 open3d 需要 'GLIBC_2.27' 文件的支持， Ubuntu 16.04 系统中缺失该文件，且该文件仅存在于 Ubuntu 18.04 及之后的系统中。

- 如果您在 `import pycocotools` 时遇到版本错误的问题，这是由于 nuscenes-devkit 需要安装 pycocotools，然而 mmdet 依赖于 mmpycocotools，当前的解决方案如下所示，我们将会在之后全面支持 pycocotools ：

  ```shell
  pip uninstall pycocotools mmpycocotools
  pip install mmpycocotools
  ```

  **注意**： 我们已经在 0.13.0 及之后的版本中全面支持 pycocotools。

- 如果您在导入 pycocotools 相关包时遇到下面的问题：

  `ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject`

  请将 pycocotools 的版本降级至 2.0.1，这是由于最新版本的 pycocotools 与 numpy \< 1.20.0 不兼容。或者通过下面的方式从源码进行编译来安装最新版本的 pycocotools ：

  `pip install -e "git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI"`

  或者

  `pip install -e "git+https://github.com/ppwwyyxx/cocoapi#egg=pycocotools&subdirectory=PythonAPI"`

- 如果您使用 cuda-9.0 的环境并遇到关于 numba 的错误， 您应该检查下 numba 的版本。在 cuda-9.0 环境中，高版本的 numba 是不支持的，我们建议安装 numba==0.53.0.

## 如何标注点云？

MMDetection3D 不支持点云标注。我们提供一些开源的标注工具供参考：

- [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS)
- [LATTE](https://github.com/bernwang/latte)

此外，我们改进了 [LATTE](https://github.com/bernwang/latte) 以便更方便的标注。更多的细节请参考[这里](https://arxiv.org/abs/2011.10174)。
