# Waymo Dataset

This page provides specific tutorials about the usage of MMDetection3D for Waymo dataset.

## Prepare dataset

Before preparing Waymo dataset, if you only installed requirements in `requirements/build.txt` and `requirements/runtime.txt` before, please install the official package for this dataset at first by running

```
# tf 2.1.0.
pip install waymo-open-dataset-tf-2-1-0==1.2.0
# tf 2.0.0
# pip install waymo-open-dataset-tf-2-0-0==1.2.0
# tf 1.15.0
# pip install waymo-open-dataset-tf-1-15-0==1.2.0
```

or

```
pip install -r requirements/optional.txt
```

Like the general way to prepare dataset, it is recommended to symlink the dataset root to `$MMDETECTION3D/data`.
Due to the original Waymo data format is based on `tfrecord`, we need to preprocess the raw data for convenient usage in the training and evaluation procedure. Our approach is to convert them into KITTI format.

The folder structure should be organized as follows before our processing.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── waymo
│   │   ├── waymo_format
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
│   │   │   ├── gt.bin
│   │   ├── kitti_format
│   │   │   ├── ImageSets

```

You can download Waymo open dataset V1.2 [HERE](https://waymo.com/open/download/) and its data split [HERE](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o?usp=sharing). Then put `tfrecord` files into corresponding folders in `data/waymo/waymo_format/` and put the data split txt files into `data/waymo/kitti_format/ImageSets`. Download ground truth bin files for validation set [HERE](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0/validation/ground_truth_objects) and put it into `data/waymo/waymo_format/`. A tip is that you can use `gsutil` to download the large-scale dataset with commands. You can take this [tool](https://github.com/RalphMao/Waymo-Dataset-Tool) as an example for more details. Subsequently, prepare Waymo data by running

```bash
python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --workers 128 --extra-tag waymo
```

Note that if your local disk does not have enough space for saving converted data, you can change the `--out-dir` to anywhere else. Just remember to create folders and prepare data there in advance and link them back to `data/waymo/kitti_format` after the data conversion.

After the data conversion, the folder structure and info files should be organized as below.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── waymo
│   │   ├── waymo_format
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
│   │   │   ├── gt.bin
│   │   ├── kitti_format
│   │   │   ├── ImageSets
│   │   │   ├── training
│   │   │   │   ├── calib
│   │   │   │   ├── image_0
│   │   │   │   ├── image_1
│   │   │   │   ├── image_2
│   │   │   │   ├── image_3
│   │   │   │   ├── image_4
│   │   │   │   ├── label_0
│   │   │   │   ├── label_1
│   │   │   │   ├── label_2
│   │   │   │   ├── label_3
│   │   │   │   ├── label_4
│   │   │   │   ├── label_all
│   │   │   │   ├── pose
│   │   │   │   ├── velodyne
│   │   │   ├── testing
│   │   │   │   ├── (the same as training)
│   │   │   ├── waymo_gt_database
│   │   │   ├── waymo_infos_trainval.pkl
│   │   │   ├── waymo_infos_train.pkl
│   │   │   ├── waymo_infos_val.pkl
│   │   │   ├── waymo_infos_test.pkl
│   │   │   ├── waymo_dbinfos_train.pkl

```

Here because there are several cameras, we store the corresponding image and labels that can be projected to that camera respectively and save pose for further usage of consecutive frames point clouds. We use a coding way `{a}{bbb}{ccc}` to name the data for each frame, where `a` is the prefix for different split (`0` for training, `1` for validation and `2` for testing), `bbb` for segment index and `ccc` for frame index. You can easily locate the required frame according to this naming rule. We gather the data for training and validation together as KITTI and store the indices for different set in the `ImageSet` files.

## Training

Considering there are many similar frames in the original dataset, we can basically use a subset to train our model primarily. In our preliminary baselines, we load one frame every five frames, and thanks to our hyper parameters settings and data augmentation, we obtain a better result compared with the performance given in the original dataset [paper](https://arxiv.org/pdf/1912.04838.pdf). For more details about the configuration and performance, please refer to README.md in the `configs/pointpillars/`. A more complete benchmark based on other settings and methods is coming soon.

## Evaluation

For evaluation on Waymo, please follow the [instruction](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md) to build the binary file `compute_detection_metrics_main` for metrics computation and put it into `mmdet3d/core/evaluation/waymo_utils/`.  Basically, you can follow the commands below to install `bazel` and build the file.

```shell
# download the code and enter the base directory
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
# git clone https://github.com/Abyssaledge/waymo-open-dataset-master waymo-od # if you want to use faster multi-thread version.
cd waymo-od
git checkout remotes/origin/master

# use the Bazel build system
sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=3.1.0
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential

# configure .bazelrc
./configure.sh
# delete previous bazel outputs and reset internal caches
bazel clean

bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main ../mmdetection3d/mmdet3d/evaluation/functional/waymo_utils/
```

Then you can evaluate your models on Waymo. An example to evaluate PointPillars on Waymo with 8 GPUs with Waymo metrics is as follows.

```shell
./tools/dist_test.sh configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car.py checkpoints/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car_latest.pth
```

`pklfile_prefix` should be set in `test_evaluator` of configuration if the bin file is needed to be generated, so you can add `--cfg-options "test_evaluator.pklfile_prefix=xxxx"` in the end of command if you want do it.

**Notice**:

1. Sometimes when using `bazel` to build `compute_detection_metrics_main`, an error `'round' is not a member of 'std'` may appear. We just need to remove the `std::` before `round` in that file.

2. Considering it takes a little long time to evaluate once, we recommend to evaluate only once at the end of model training.

3. To use TensorFlow with CUDA 9, it is recommended to compile it from source. Apart from official tutorials, you can refer to this [link](https://github.com/SmileTM/Tensorflow2.X-GPU-CUDA9.0) for possibly suitable precompiled packages and useful information for compiling it from source.

## Testing and make a submission

An example to test PointPillars on Waymo with 8 GPUs, generate the bin files and make a submission to the leaderboard.

`submission_prefix` should be set in `test_evaluator` of configuration before you run the test command if you want to generate the bin files and make a submission to the leaderboard..

After generating the bin file, you can simply build the binary file `create_submission` and use them to create a submission file by following the [instruction](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md/). Basically, here are some example commands.

```shell
cd ../waymo-od/
bazel build waymo_open_dataset/metrics/tools/create_submission
cp bazel-bin/waymo_open_dataset/metrics/tools/create_submission ../mmdetection3d/mmdet3d/core/evaluation/waymo_utils/
vim waymo_open_dataset/metrics/tools/submission.txtpb  # set the metadata information
cp waymo_open_dataset/metrics/tools/submission.txtpb ../mmdetection3d/mmdet3d/evaluation/functional/waymo_utils/

cd ../mmdetection3d
# suppose the result bin is in `results/waymo-car/submission`
mmdet3d/core/evaluation/waymo_utils/create_submission  --input_filenames='results/waymo-car/kitti_results_test.bin' --output_filename='results/waymo-car/submission/model' --submission_filename='mmdet3d/evaluation/functional/waymo_utils/submission.txtpb'

tar cvf results/waymo-car/submission/my_model.tar results/waymo-car/submission/my_model/
gzip results/waymo-car/submission/my_model.tar
```

For evaluation on the validation set with the eval server, you can also use the same way to generate a submission. Make sure you change the fields in `submission.txtpb` before running the command above.
