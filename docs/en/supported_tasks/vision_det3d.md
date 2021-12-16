# Vision-Based 3D Detection

Vision-based 3D detection refers to the 3D detection solutions based on vision-only input, such as monocular, binocular, and multi-view image based 3D detection.
Currently, we only support monocular and multi-view 3D detection methods. Other approaches should be also compatible with our framework and will be supported in the future.

It expects the given model to take any number of images as input, and predict the 3D bounding boxes and category labels for each object of interest.
Taking FCOS3D on the nuScenes dataset as an example, we will show how to prepare data, train and test a model on a standard 3D detection benchmark, and how to visualize and validate the results.

## Data Preparation

To begin with, we need to download the raw data and reorganize the data in a standard way presented in the [doc for data preparation](https://mmdetection3d.readthedocs.io/en/latest/data_preparation.html).

Due to different ways of organizing the raw data in different datasets, we typically need to collect the useful data information with a .pkl or .json file.
So after getting all the raw data ready, we need to run the scripts provided in the `create_data.py` for different datasets to generate data infos.
For example, for nuScenes we need to run:

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

Afterwards, the related folder structure should be as follows:

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_trainval.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
│   │   ├── nuscenes_infos_train_mono3d.coco.json
│   │   ├── nuscenes_infos_trainval_mono3d.coco.json
│   │   ├── nuscenes_infos_val_mono3d.coco.json
│   │   ├── nuscenes_infos_test_mono3d.coco.json
```

Note that the .pkl files here are mainly used for methods using LiDAR data and .json files are used for 2D detection/vision-only 3D detection.
The .json files only contain infos for 2D detection before supporting monocular 3D detection in v0.13.0, so if you need the latest infos, please checkout the branches after v0.13.0.

## Training

Then let us train a model with provided configs for FCOS3D. The basic script is the same as other models.
You can basically follow the examples provided in this [tutorial](https://mmdetection3d.readthedocs.io/en/latest/1_exist_data_model.html#inference-with-existing-models) when training with different GPU settings.
Suppose we use 8 GPUs on a single machine with distributed training:

```
./tools/dist_train.sh configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py 8
```

Note that `2x8` in the config name refers to the training is completed with 8 GPUs and 2 data samples on each GPU.
If your customized setting is different from this, sometimes you need to adjust the learning rate accordingly.
A basic rule can be referred to [here](https://arxiv.org/abs/1706.02677).

We can also achieve better performance with finetuned FCOS3D by running:

```
./tools/dist_train.sh fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py 8
```

after training a baseline model with the previous script.
Please remember to modify the path [here](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py#L8) correspondingly.

## Quantitative Evaluation

During training, the model checkpoints will be evaluated regularly according to the setting of `evaluation = dict(interval=xxx)` in the config.

We support official evaluation protocols for different datasets.
Due to the output format is the same as 3D detection based on other modalities, the evaluation methods are also the same.

For nuScenes, the model will be evaluated with distance-based mean AP (mAP) and NuScenes Detection Score (NDS) for 10 categories respectively.
The evaluation results will be printed in the command like:

```
mAP: 0.3197
mATE: 0.7595
mASE: 0.2700
mAOE: 0.4918
mAVE: 1.3307
mAAE: 0.1724
NDS: 0.3905
Eval time: 170.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.503   0.577   0.152   0.111   2.096   0.136
truck   0.223   0.857   0.224   0.220   1.389   0.179
bus     0.294   0.855   0.204   0.190   2.689   0.283
trailer 0.081   1.094   0.243   0.553   0.742   0.167
construction_vehicle    0.058   1.017   0.450   1.019   0.137   0.341
pedestrian      0.392   0.687   0.284   0.694   0.876   0.158
motorcycle      0.317   0.737   0.265   0.580   2.033   0.104
bicycle 0.308   0.704   0.299   0.892   0.683   0.010
traffic_cone    0.555   0.486   0.309   nan     nan     nan
barrier 0.466   0.581   0.269   0.169   nan     nan
```

In addition, you can also evaluate a specific model checkpoint after training is finished. Simply run scripts like the following:

```
./tools/dist_test.sh configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py \
    work_dirs/fcos3d/latest.pth --eval mAP
```

## Testing and Making a Submission

If you would like to only conduct inference or test the model performance on the online benchmark,
you just need to replace the `--eval mAP` with `--format-only` in the previous evaluation script and specify the `jsonfile_prefix` if necessary,
e.g., adding an option `--eval-options jsonfile_prefix=work_dirs/fcos3d/test_submission`.
Please guarantee the [info for testing](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/datasets/nus-mono3d.py#L93) in the config corresponds to the test set instead of validation set.

After generating the results, you can basically compress the folder and upload to the evalAI evaluation server for nuScenes 3D detection challenge.

## Qualitative Validation

MMDetection3D also provides versatile tools for visualization such that we can have an intuitive feeling of the detection results predicted by our trained models.
You can either set the `--eval-options 'show=True' 'out_dir=${SHOW_DIR}'` option to visualize the detection results online during evaluation,
or using `tools/misc/visualize_results.py` for offline visualization.

Besides, we also provide scripts `tools/misc/browse_dataset.py` to visualize the dataset without inference.
Please refer more details in the [doc for visualization](https://mmdetection3d.readthedocs.io/en/latest/useful_tools.html#visualization).

Note that currently we only support the visualization on images for vision-only methods.
The visualization in the perspective view and bird-eye-view (BEV) will be integrated in the future.
