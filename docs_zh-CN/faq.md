# 常见问题解答

我们列出了一些用户和开发者在开发过程中会遇到的常见问题以及对应的解决方案，如果您发现了任何频繁出现的问题，请随时扩充本列表，非常欢迎您提出的任何解决方案。如果您在环境配置、模型训练等工作中遇到任何的问题，请使用[问题模板](https://github.com/open-mmlab/mmdetection3d/blob/master/.github/ISSUE_TEMPLATE/error-report.md/)来创建相应的 issue,并将所需的所有信息填入到问题模板中，我们会尽快解决您的问题。

## MMCV/MMDet/MMDet3D Installation

- 如果您在 `import open3d` 时遇到下面的问题：

  ``OSError: /lib/x86_64-linux-gnu/libm.so.6: version 'GLIBC_2.27' not found``

  请将 open3d 的版本降级至 0.9.0.0，因为最新版 open3d 需要 'GLIBC_2.27' 文件的支持， Ubuntu 16.04 系统中缺失该文件，且该文件仅存在于 Ubuntu 18.04 及之后的系统中。

- 如果您在 `import pycocotools` 时遇到版本错误的问题，这是由于 nuscenes-devkit 需要安装 pycocotools，然而 mmdet 依赖于 mmpycocotools，当前的解决方案如下所示，我们将会在之后全面支持 pycocotools ：

  ```shell
  pip uninstall pycocotools mmpycocotools
  pip install mmpycocotools
  ```

  **注意**： 我们已经在 0.13.0 及之后的版本中全面支持 pycocotools。

- 如果您遇到下面的问题，并且您的环境包含 numba == 0.48.0 和 numpy >= 1.20.0：

  ``TypeError: expected dtype object, got 'numpy.dtype[bool_]'``

    请将 numpy 的版本降级至 < 1.20.0，或者从源码安装 numba == 0.48，这是由于 numpy == 1.20.0 改变了 API，使得在调用 `np.dtype` 会产生子类。请参考 [这里](https://github.com/numba/numba/issues/6041) 获取更多细节。
