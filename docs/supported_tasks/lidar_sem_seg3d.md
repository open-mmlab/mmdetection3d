# LiDAR-Based 3D Semantic Segmentation

LiDAR-based 3D semantic segmentation is one of the most basic tasks supported in MMDetection3D.
It expects the given model to take any number of points with features collected by LiDAR as input, and predict the semantic labels for each input point.
Next, taking PointNet++ (SSG) on the ScanNet dataset as an example, we will show how to prepare data, train and test a model on a standard 3D semantic segmentation benchmark, and how to visualize and validate the results.

## Data Preparation

To begin with, we need to download the raw data from ScanNet's [official website](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation).

Due to different ways of organizing the raw data in different datasets, we typically need to collect the useful data information with a .pkl or .json file.

So after getting all the raw data ready, we can follow the instructions presented in [ScanNet README doc](https://github.com/open-mmlab/mmdetection3d/blob/master/data/scannet/README.md/) to generate data infos.

Afterwards, the related folder structure should be as follows:

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── scannet
│   │   ├── scannet_utils.py
│   │   ├── batch_load_scannet_data.py
│   │   ├── load_scannet_data.py
│   │   ├── scannet_utils.py
│   │   ├── README.md
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── scannet_instance_data
│   │   ├── points
│   │   ├── instance_mask
│   │   ├── semantic_mask
│   │   ├── seg_info
│   │   │   ├── train_label_weight.npy
│   │   │   ├── train_resampled_scene_idxs.npy
│   │   │   ├── val_label_weight.npy
│   │   │   ├── val_resampled_scene_idxs.npy
│   │   ├── scannet_infos_train.pkl
│   │   ├── scannet_infos_val.pkl
│   │   ├── scannet_infos_test.pkl
```

## Training

Then let us train a model with provided configs for PointNet++ (SSG).
You can basically follow this [tutorial](https://mmdetection3d.readthedocs.io/en/latest/1_exist_data_model.html#inference-with-existing-models) for sample scripts when training with different GPU settings.
Suppose we use 2 GPUs on a single machine with distributed training:

```
./tools/dist_train.sh configs/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py 2
```

Note that `16x2` in the config name refers to the training is completed with 2 GPUs and 16 samples on each GPU.
If your customized setting is different from this, sometimes you need to adjust the learning rate accordingly.
A basic rule can be referred to [here](https://arxiv.org/abs/1706.02677).

## Quantitative Evaluation

During training, the model checkpoints will be evaluated regularly according to the setting of `evaluation = dict(interval=xxx)` in the config.
We support official evaluation protocols for different datasets.
For ScanNet, the model will be evaluated with mean Intersection over Union (mIoU) over all 20 categories.
The evaluation results will be printed in the command like:

```
+---------+--------+--------+---------+--------+--------+--------+--------+--------+--------+-----------+---------+---------+--------+---------+--------------+----------------+--------+--------+---------+----------------+--------+--------+---------+
| classes | wall   | floor  | cabinet | bed    | chair  | sofa   | table  | door   | window | bookshelf | picture | counter | desk   | curtain | refrigerator | showercurtrain | toilet | sink   | bathtub | otherfurniture | miou   | acc    | acc_cls |
+---------+--------+--------+---------+--------+--------+--------+--------+--------+--------+-----------+---------+---------+--------+---------+--------------+----------------+--------+--------+---------+----------------+--------+--------+---------+
| results | 0.7257 | 0.9373 | 0.4625  | 0.6613 | 0.7707 | 0.5562 | 0.5864 | 0.4010 | 0.4558 | 0.7011    | 0.2500  | 0.4645  | 0.4540 | 0.5399  | 0.2802       | 0.3488         | 0.7359 | 0.4971 | 0.6922  | 0.3681         | 0.5444 | 0.8118 | 0.6695  |
+---------+--------+--------+---------+--------+--------+--------+--------+--------+--------+-----------+---------+---------+--------+---------+--------------+----------------+--------+--------+---------+----------------+--------+--------+---------+
```

In addition, you can also evaluate a specific model checkpoint after training is finished. Simply run scripts like the following:

```
./tools/dist_test.sh configs/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py \
    work_dirs/pointnet2_ssg/latest.pth --eval mIoU
```

## Testing and Making a Submission

If you would like to only conduct inference or test the model performance on the online benchmark,
you need to replace the `--eval mIoU` with `--format-only` in the previous evaluation script and change `ann_file=data_root + 'scannet_infos_val.pkl'` to `ann_file=data_root + 'scannet_infos_test.pkl'` in the ScanNet dataset's [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/scannet_seg-3d-20class.py#L126). Remember to specify the `txt_prefix` as the directory to save the testing results,
e.g., adding an option `--eval-options txt_prefix=work_dirs/pointnet2_ssg/test_submission`.
After generating the results, you can basically compress the folder and upload to the [ScanNet evaluation server](http://kaldir.vc.in.tum.de/scannet_benchmark/semantic_label_3d).

## Qualitative Validation

MMDetection3D also provides versatile tools for visualization such that we can have an intuitive feeling of the segmentation results predicted by our trained models.
You can either set the `--eval-options 'show=True' 'out_dir=${SHOW_DIR}'` option to visualize the segmentation results online during evaluation,
or using `tools/misc/visualize_results.py` for offline visualization.
Besides, we also provide scripts `tools/misc/browse_dataset.py` to visualize the dataset without inference.
Please refer more details in the [doc for visualization](https://mmdetection3d.readthedocs.io/en/latest/useful_tools.html#visualization).
