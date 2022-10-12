# 3D 语义分割 SensatUrban 数据集

## 数据集的准备

由于Urban数据集单个文件较大，我们提供了下列多种生成数据集的方式，请根据需要选择适合的方式：

- 3D点云切片数据集

  - [x] 方形滑动窗口切割方式
  - [x] 方形随机切割方式
  - [ ] 最近邻随机切割方式

- 3D点云降采样数据集

  - [x] 随机降采样方式
  - [x] 均匀降采样方式
  - [ ] 体素降采样方式，暂未支持，请参照[官方处理方法](https://github.com/QingyongHu/SensatUrban/blob/master/input_preparation.py)

在3D点云切片数据集基础上，考虑到多模态融合或其他用途，可以选择同时生成2D语义分割数据集：

- 2D语义分割数据集（切片形状：方形）

  - [x] 深度图数据集
  - [x] RGB图像数据集

可以通过设置，任意组合生成以上数据集类型，并且支持如下文件结构组织风格。

- [x] Potsdam

- [ ] SemanticKITTI

## 准备SensatUrban数据

下载并解压后的`data_release`请按照如下结构放置在相应目录下。

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── sensaturban
│   │   ├── train
│   │   │   ├── birmingham_block_0.ply
│   │   │   ├── xxxxx_block_x.ply
│   │   ├── test
│   │   │   ├── birmingham_block_2.ply
│   │   │   ├── xxxxx_block_x.ply
```

SensatUrban数据集有37个train文件和6个test文件，每个ply文件有`x,y,z,red,green,blue,class`7个数据域，一共有13个类别，他们分别是:

`0: 'Ground', 1: 'High Vegetation', 2: 'Buildings', 3: 'Walls', 4: 'Bridge', 5: 'Parking', 6: 'Rail', 7: 'traffic Roads', 8: 'Street Furniture',  9: 'Cars', 10: 'Footpath', 11: 'Bikes', 12: 'Water'`

## 创建数据集

由于python多线程的问题，建议使用如下的方式对数据进行处理以加快处理速度。
首先创建一个`create_sensaturban_dataset.py`

```python
import argparse
from sensaturban_converter import SensatUrbanConverter
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=0, type=int)
    converter = SensatUrbanConverter(
            root_path='./', # 原始数据集的路径
            info_prefix='sensaturban', # info文件的前缀
            out_dir='./', # 输出数据集的路径
            workers=1, # 同时处理数据集的线程数
            to_image=False, # 是否生成2D数据集,如果为true则必须指定切片方式
            subsample_method='none', # 是否生成以及如何生成降采样数据集
            crop_method='sliding', # 是否生成切片数据集以及如何生成切片数据集
            crop_size=12.5, # 切片数据集的边长为2 * crop_size，与crop_method一起使用
            crop_scale=0.05, # 2D数据集中，每个像素代表实际多少米
            subsample_rate=0.5, # 降采样数据集中的参数，当随机随机降采样时输入为点，当体素降采样时为体素大小
            random_crop_ratio=1.0, # 随机切片方式时，根据文件大小计算切割次数，默认每MB切 random_crop_ratio次
        )
    args, opts = parser.parse_known_args()
    converter._convert2potsdam_one(args.id)
```

然后就可以通过执行如下命令并行处理，请根据运行设备的配置情况自行选择同时处理的数量

```shell
python create_sensaturban_dataset.py --id 0 &
python create_sensaturban_dataset.py --id 1 &
...
```

与其他数据集生成方式类似，我们也可以通过`python tools/create_data.py sensaturban --root-path ./data/sensaturban --out-dir ./data/sensaturban`命令生成，但在此之前，
我们需要根据自己的需求修改`create_data.py`文件中的参数，为了更好的了解如何生成数据集，下面简单介绍一下影响生成数据集的参数：

```python
def sensaturban_data_prep(root_path,
                    info_prefix,
                    out_dir,
                    workers,
                    dataset_style='potsdam'): # 数据集的文件风格

    converter = SensatUrbanConverter(
        root_path, # 原始数据集的路径
        info_prefix, # info文件的前缀
        out_dir, # 输出数据集的路径
        workers, # 同时处理数据集的线程数
        to_image=False, # 是否生成2D数据集,如果为true则必须指定切片方式
        subsample_method='none', # 是否生成以及如何生成降采样数据集
        crop_method='sliding', # 是否生成切片数据集以及如何生成切片数据集
        crop_size=12.5, # 切片数据集的边长为2 * crop_size，与crop_method一起使用
        crop_scale=0.05, # 2D数据集中，每个像素代表实际多少米
        subsample_rate=0.5, # 降采样数据集中的参数，当随机随机降采样时输入为点，当体素降采样时为体素大小
        random_crop_ratio=1.0, # 随机切片方式时，根据文件大小计算切割次数，默认每MB切 random_crop_ratio次
    )
```

下面将提供一些具体的例子，可以根据需求进行选择：

### 3D点云切片数据集

- 滑动窗口切片方式

```python
def sensaturban_data_prep(root_path,
                    info_prefix,
                    out_dir,
                    workers,
                    dataset_style='potsdam'): # 数据集的文件风格

    converter = SensatUrbanConverter(
        root_path, # 原始数据集的路径
        info_prefix, # info文件的前缀
        out_dir, # 输出数据集的路径
        workers, # 同时处理数据集的线程数
        to_image=False, # 不生成2D数据集
        subsample_method='none', # 不生成降采样数据集
        crop_method='sliding', # 使用滑动窗口方式的切片数据集
        crop_size=12.5, # 每个切片的边长为25m x 25m
    )
```

此时生成的目录结构为：

```
data
├── sensaturban
│   ├── train
│   │   ├── points
│   │   │   ├── xxxxx.bin # 点云文件
│   │   ├── labels
│   │   │   ├── xxxxx.labels # 点云对应的点标签文件
```

- 随机切片方式

```python
def sensaturban_data_prep(root_path,
                    info_prefix,
                    out_dir,
                    workers,
                    dataset_style='potsdam'): # 数据集的文件风格

    converter = SensatUrbanConverter(
        root_path, # 原始数据集的路径
        info_prefix, # info文件的前缀
        out_dir, # 输出数据集的路径
        workers, # 同时处理数据集的线程数
        to_image=False, # 不生成2D数据集
        subsample_method='none', # 不生成降采样数据集
        crop_method='random', # 使用随机切片方式的切片数据集
        random_crop_ratio=1.0, # 一个文件中每1MB切片一次
    )
```

此时生成的目录结构为：

```
data
├── sensaturban
│   ├── train
│   │   ├── points
│   │   │   ├── xxxxx.bin # 点云文件
│   │   ├── labels
│   │   │   ├── xxxxx.labels # 点云对应的点标签文件
```

### 降采样数据集

- 随机降采样数据集

```python
def sensaturban_data_prep(root_path,
                    info_prefix,
                    out_dir,
                    workers,
                    dataset_style='potsdam'): # 数据集的文件风格

    converter = SensatUrbanConverter(
        root_path, # 原始数据集的路径
        info_prefix, # info文件的前缀
        out_dir, # 输出数据集的路径
        workers, # 同时处理数据集的线程数
        to_image=False, # 不生成2D数据集
        subsample_method='random', # 随机降采样
        crop_method='none', # 不生成切片数据集
        subsample_rate=100000, # 随机取100000个点
    )
```

此时生成的目录结构为：

```
data
├── sensaturban
│   ├── train
│   │   ├── points
│   │   │   ├── xxxxx.bin # 点云文件
│   │   ├── labels
│   │   │   ├── xxxxx.labels # 点云对应的点标签文件
```

### 3D滑动窗口切片数据集+2D语义分割数据集

```python
def sensaturban_data_prep(root_path,
                    info_prefix,
                    out_dir,
                    workers,
                    dataset_style='potsdam'): # 数据集的文件风格

    converter = SensatUrbanConverter(
        root_path, # 原始数据集的路径
        info_prefix, # info文件的前缀
        out_dir, # 输出数据集的路径
        workers, # 同时处理数据集的线程数
        to_image=True, # 生成2D语义分割数据集
        subsample_method='none', # 不生成降采样数据集
        crop_method='sliding', # 采用滑动窗口的方式生成数据集
        crop_size=12.5, # 切片点云范围大小25m x 25m
        crop_scale=0.05, # 每个像素代表0.05m，所以图像大小为500x500
    )
```

此时生成的目录结构为：

```
data
├── sensaturban
│   ├── train
│   │   ├── points
│   │   │   ├── xxxxx.bin # 点云文件
│   │   ├── rgbs
│   │   │   ├── xxxxx.png # rgb图像文件
│   │   ├── depths
│   │   │   ├── xxxxx.npy # 深度图文件
│   │   ├── masks
│   │   │   ├── xxxxx.png # 2D分割真值图像
│   │   ├── labels
│   │   │   ├── xxxxx.labels # 点云对应的点标签文件
```

## 训练流程

```python
train_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            shift_height=True,
            load_dim=6,
            use_dim=[0, 1, 2, 3, 4, 5]),
        dict(
            type='LoadImageFromFile',
            color_type='color',
            imdecode_backend='cv2'),
        dict(
            type='LoadDepthFromFile'),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True,
            with_seg=True,
            seg_3d_dtype=np.int8),
        dict(
            type='Pack3DDetInputs',
            keys=[
                'points', 'img', 'depth_img', 'pts_semantic_mask', 'gt_seg_map'
            ])
        ]
```

## 度量指标

通常我们使用平均交并比 (mean Intersection over Union, mIoU) 作为 SensatUrban 语义分割任务的度量指标。
具体而言，我们先计算所有类别的 IoU，然后取平均值作为 mIoU。
更多实现细节请参考 [seg_eval.py](https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/evaluation/seg_eval.py)。

另外提供了由2D数据集投影回3D数据集的工具类 `SensatUrbanEvaluator`，由于python多线程存在的问题，
我们建议采用如下方式进行并行的处理以加速重投影过程，
你可以通过创建一个新`reproject.py`文件并添加如下代码并根据自己数据集的情况调整参数设置：

```python
import argparse
from sensaturban_data_utils import SensatUrbanEvaluator
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=0, type=int)
    evaluater = SensatUrbanEvaluator(
        split='test',
        dataset_path='./sensaturban',
        pred_path='./pred',
        crop_method='random',
        out_path='./ply_out',
        crop_size=12.5,
        bev_size=500,
        bev_scale=0.05,
        out_ply=False,
        out_label=True)
    args, opts = parser.parse_known_args()
    evaluater.generate(args.id)
```

最后通过执行命令

```python
python reproject.py --id 0 &
python reproject.py --id 1 &
python reproject.py --id 2 &
```

来并行处理重投影，注意随机方式的重投影需要耗费较大的内存，请根据运行设备情况选择执行多少进程。
