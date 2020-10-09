# Tutorial 5: Waymo Dataset

This page provides specific tutorials about the usage of MMDetection3D for waymo dataset.

## Prepare datasets

Before preparing waymo dataset, if you only installed requirements in `requirements/build.txt` and `requirements/runtime.txt` before, please install the official package for this dataset at first by running

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
Due to the original waymo data format is based on `tfrecord`, we need to preprocess the raw data for convenient usage in the training and evaluation procedure. Our approach is to convert them into KITTI format.

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

You can download Waymo open dataset V1.2 [HERE](https://waymo.com/open/download/) and its data split [HERE](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o?usp=sharing). Then put tfrecord files into corresponding folders in `data/waymo/waymo_format/` and put the data split txt files into `data/waymo/kitti_format/ImageSets`. Download ground truth bin file for validation set [HERE](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0/validation/ground_truth_objects) and put it into `data/waymo/waymo_format/`. A tip is that you can use `gsutil` to download the large-scale dataset with commands. You can take this [tool](https://github.com/RalphMao/Waymo-Dataset-Tool) as an example for more details. Subsequently, prepare waymo data by running

```bash
python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --workers 128 --extra-tag waymo
```

Note that if your local disk does not have enough space for saving converted data, you can change the `out-dir` to anywhere else. Just remember to create folders and prepare data there in advance and link them back to `data/waymo/kitti_format` after the data conversion.

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

Here because there are several cameras, we store the corresponding image and labels that can be projected to that camera respectively and save pose for further usage of consecutive frames point clouds. We use a coding way `{a}{bbb}{ccc}` to name the data for each frame, where `a` is the prefix for different split (`0` for training, `1` for validation and `2` for testing), `bbb` for segment index and `ccc` for frame index. You can easily locate the required frame according to this naming rule. We gather the data for training and validation together as KITTI and store the indices for different set in the ImageSet files.

## Training

Considering there are many similar frames in the original dataset, we can basically use a subset to train our model primarily. In our preliminary baselines, we load one frame every five frames, and thanks to our hyper parameters settings and data augmentation, we obtain a better result compared with the performance given in the original dataset [paper](https://arxiv.org/pdf/1912.04838.pdf). For more details about the configuration and performance, please refer to README.md in the `configs/pointpillars/`. A more complete benchmark based on other settings and methods is coming soon.

## Evaluation

For evaluation on waymo, please follow the [instruction](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md) to build the binary file `compute_detection_metrics_main` for metrics computation and put it into `mmdet3d/core/evaluation/waymo_utils/`.  Basically, you can follow the commands below to install bazel and build the file.

   ```shell
   git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
   cd waymo-od
   git checkout remotes/origin/master

   sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
   wget https://github.com/bazelbuild/bazel/releases/download/0.28.0/bazel-0.28.0-installer-linux-x86_64.sh
   sudo bash bazel-0.28.0-installer-linux-x86_64.sh
   sudo apt install build-essential

   ./configure.sh
   bazel clean

   bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
   cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main ../mmdetection3d/mmdet3d/core/evaluation/waymo_utils/
   ```

Then you can evaluate your models on waymo. An example to evaluate PointPillars on waymo with 8 GPUs with waymo metrics is as follows.

   ```shell
   ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} configs/pointpillars/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car.py \
       checkpoints/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car_latest.pth --out results/waymo-car/results_eval.pkl \
       --eval waymo --options 'pklfile_prefix=results/waymo-car/kitti_results' \
       'submission_prefix=results/waymo-car/kitti_results'
   ```

`pklfile_prefix` should be given in the options if the bin file is needed to be generated. For metrics, `waymo` is the recommended official evaluation prototype. Currently, evaluating with choice `kitti` is adapted from KITTI and the results for each difficulty are not exactly the same as the definition of KITTI. Instead, most of objects are marked with difficulty 0 currently, which will be fixed in the future. The reasons of its instability include the large computation for evalution, the lack of occlusion and truncation in the converted data, different definition of difficulty and different methods of computing average precision.

**Notice**:

1. Sometimes when using bazel to build `compute_detection_metrics_main`, an error `'round' is not a member of 'std'` may appear. We just need to remove the `std::` before `round` in that file.

2. Considering it takes a little long time to evaluate once, we recommend to evaluate only once at the end of model training.

3. To use tensorflow with cuda9, it is recommended to compile it from source. Apart from official tutorials, you can refer to this [link](https://github.com/SmileTM/Tensorflow2.X-GPU-CUDA9.0) for possibly suitable precompiled packages and useful information for compiling it from source.

## Testing and make a submission

An example to test PointPillars on waymo with 8 GPUs, generate the bin files and make a submission to the leaderboard.

   ```shell
   ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} configs/pointpillars/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car.py \
       checkpoints/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car_latest.pth --out results/waymo-car/results_eval.pkl \
       --format-only --options 'pklfile_prefix=results/waymo-car/kitti_results' \
       'submission_prefix=results/waymo-car/kitti_results'
   ```

After generating the bin file, you can simply build the binary file `create_submission` and use them to create a submission file by following the [instruction](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md). Basically, here are some example commands.

   ```shell
   cd ../waymo-od/
   bazel build waymo_open_dataset/metrics/tools/create_submission
   cp bazel-bin/waymo_open_dataset/metrics/tools/create_submission ../mmdetection3d/mmdet3d/core/evaluation/waymo_utils/
   vim waymo_open_dataset/metrics/tools/submission.txtpb  # set the metadata information
   cp waymo_open_dataset/metrics/tools/submission.txtpb ../mmdetection3d/mmdet3d/core/evaluation/waymo_utils/

   cd ../mmdetection3d
   # suppose the result bin is in `results/waymo-car/submission`
   mmdet3d/core/evaluation/waymo_utils/create_submission  --input_filenames='results/waymo-car/kitti_results_test.bin' --output_filename='results/waymo-car/submission/model' --submission_filename='mmdet3d/core/evaluation/waymo_utils/submission.txtpb'

   tar cvf results/waymo-car/submission/my_model.tar results/waymo-car/submission/my_model/
   gzip results/waymo-car/submission/my_model.tar
   ```

For evaluation on the validation set with the eval server, you can also use the same way to generate a submission. Make sure you change the fields in submission.txtpb before running the command above.
