### Prepare SUN RGB-D Data
We follow the procedure in [votenet](https://github.com/facebookresearch/votenet/).

1. Download SUNRGBD v2 data [HERE](http://rgbd.cs.princeton.edu/data/). Then, move SUNRGBD.zip, SUNRGBDMeta2DBB_v2.mat, SUNRGBDMeta3DBB_v2.mat and SUNRGBDtoolbox.zip to the OFFICIAL_SUNRGBD folder, unzip the zip files.

2. Enter the `matlab` folder, run `extract_split.m`, `extract_rgbd_data_v2.m` and `extract_rgbd_data_v1.m` to extract point clouds and annotations.

3. Back to this level, prepare data by running `python sunrgbd_data.py --gen_v1_data`

4. Enter the project root directory, generate training data by running `python tools/create_data.py sunrgbd --root-path ./data/sunrgbd --out-dir ./data/sunrgbd --extra-tag sunrgbd`.

NOTE: SUNRGBDtoolbox.zip should have MD5 hash `18d22e1761d36352f37232cba102f91f` (you can check the hash with `md5 SUNRGBDtoolbox.zip` on Mac OS or `md5sum SUNRGBDtoolbox.zip` on Linux)
