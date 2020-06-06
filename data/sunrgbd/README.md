### Prepare SUN RGB-D Data
We follow the procedure in [votenet](https://github.com/facebookresearch/votenet/).

1. Download SUNRGBD v2 data [HERE](http://rgbd.cs.princeton.edu/data/) (SUNRGBD.zip, SUNRGBDMeta2DBB_v2.mat, SUNRGBDMeta3DBB_v2.mat) and the toolkits (SUNRGBDtoolbox.zip). Move all the downloaded files under OFFICIAL_SUNRGBD. Unzip the zip files.

2. Extract point clouds and annotations (class, v2 2D -- xmin,ymin,xmax,ymax, and 3D bounding boxes -- centroids, size, 2D heading) by running `extract_split.m`, `extract_rgbd_data_v2.m` and `extract_rgbd_data_v1.m` under the `matlab` folder.

3. Prepare data by running `python sunrgbd_data.py --gen_v1_data`

4. Enter the project root directory, generate training data by running `python tools/create_data.py sunrgbd --root-path ./data/sunrgbd --out-dir ./data/sunrgbd --extra-tag sunrgbd`.

NOTE: SUNRGBDtoolbox.zip should have MD5 hash `18d22e1761d36352f37232cba102f91f` (you can check the hash with `md5 SUNRGBDtoolbox.zip` on Mac OS or `md5sum SUNRGBDtoolbox.zip` on Linux)
