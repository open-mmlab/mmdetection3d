# Visualization

MMDetection3D provides a `Det3DLocalVisualizer` to visualize and store the state of the model during training and testing, as well as results, with the following features.

1. Support the basic drawing interface for multi-modality data and multi tasks.
2. Support multiple backends such as local, TensorBoard, to write training status such as `loss`, `lr`, or performance evaluation metrics and to a specified single or multiple backends.
3. Support ground truth visualization on multimodal data, and cross-modal visualization of 3D detection results.

## Basic Drawing Interface

Inherited from `DetLocalVisualizer`, `Det3DLocalVisualizer` provides an interface for drawing common objects on 2D images, such as drawing detection boxes, points, text, lines, circles, polygons, and binary masks. More details about 2D drawing can refer to the visualization documentation in MMDetection. Here we introduce the 3D drawing interface:

### Drawing 3D Boxes on Point Cloud

We support drawing 3D boxes on point cloud by using `draw_bboxes_3d`.

```python
import torch
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

points = np.fromfile('tests/data/kitti/training/velodyne/000000.bin', dtype=np.float32)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points)
bboxes_3d = LiDARInstance3DBoxes(torch.tensor(
                [[8.7314, -1.8559, -1.5997, 1.2000, 0.4800, 1.8900,
                  -1.5808]])),
# Draw 3D bboxes
visualizer.draw_bboxes_3d(bboxes_3d)
visualizer.show()
```

### Drawing Projected 3D Boxes on Image

We support drawing projected 3D boxes on image by using `draw_proj_bboxes_3d`.

```python
import torch
import mmcv
import numpy as np
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

image = mmcv.imread('tests/data/kitti/training/image_2/000000.png', channel_order='rgb')
visualizer = Det3DLocalVisualizer()
# set image in visualizer
visualizer.set_image(image=image)
bboxes_3d = LiDARInstance3DBoxes(torch.tensor(
                [[8.7314, -1.8559, -1.5997, 1.2000, 0.4800, 1.8900,
                  -1.5808]])),
# `lidar2img` is needed to project 3D bboxes to image
input_meta = {'lidar2img': np.array(
        [[5.23289349e+02, 3.68831943e+02, 6.10469439e+01],
         [1.09560138e+02, 1.97404735e+02, -5.47377738e+02],
         [1.25930002e-02, 9.92229998e-01, -1.23769999e-01]])}
# Draw projected 3D bboxes on image
visualizer.draw_proj_bboxes_3d(bboxes_3d, input_meta)
visualizer.show()
```

### Drawing 3D Semantic Mask

We support draw segmentation mask via per-point colorization by using `draw_seg_mask`.

```python
import torch
from mmdet3d.visualization import Det3DLocalVisualizer

points = np.fromfile('tests/data/s3dis/points/Area_1_office_2.bin', dtype=np.float32)
points = points.reshape(-1, 3)
visualizer = Det3DLocalVisualizer()
mask = np.random.rand(points.shape[0], 3)
points_with_mask = np.concatenate((points, mask), axis=-1)
# Draw 3D points with mask
visualizer.draw_seg_mask(points_with_mask)
visualizer.show()
```

## Results

To see the prediction results of trained models, you can run the following command

```bash
python tools/test.py ${CONFIG_FILE} ${CKPT_PATH} --show --show-dir ${SHOW_DIR}
```

After running this command, plotted results including input data and the output of networks visualized on the input (e.g. `***_points.obj` and `***_pred.obj` in single-modality 3D detection task) will be saved in `${SHOW_DIR}`.

To see the prediction results during evaluation, you can run the following command

```bash
python tools/test.py ${CONFIG_FILE} ${CKPT_PATH} --eval 'mAP' --eval-options 'show=True' 'out_dir=${SHOW_DIR}'
```

After running this command, you will obtain the input data, the output of networks and ground-truth labels visualized on the input (e.g. `***_points.obj`, `***_pred.obj`, `***_gt.obj`, `***_img.png` and `***_pred.png` in multi-modality detection task) in `${SHOW_DIR}`. When `show` is enabled, [Open3D](http://www.open3d.org/) will be used to visualize the results online. If you are running test in remote server without GUI, the online visualization is not supported, you can set `show=False` to only save the output results in `{SHOW_DIR}`.

As for offline visualization, you will have two options.
To visualize the results with `Open3D` backend, you can run the following command

```bash
python tools/misc/visualize_results.py ${CONFIG_FILE} --result ${RESULTS_PATH} --show-dir ${SHOW_DIR}
```

![](../../resources/open3d_visual.*)

Or you can use 3D visualization software such as the [MeshLab](http://www.meshlab.net/) to open these files under `${SHOW_DIR}` to see the 3D detection output. Specifically, open `***_points.obj` to see the input point cloud and open `***_pred.obj` to see the predicted 3D bounding boxes. This allows the inference and results generation to be done in remote server and the users can open them on their host with GUI.

**Notice**: The visualization API is a little unstable since we plan to refactor these parts together with MMDetection in the future.

## Dataset

We also provide scripts to visualize the dataset without inference. You can use `tools/misc/browse_dataset.py` to show loaded data and ground-truth online and save them on the disk. Currently we support single-modality 3D detection and 3D segmentation on all the datasets, multi-modality 3D detection on KITTI and SUN RGB-D, as well as monocular 3D detection on nuScenes. To browse the KITTI dataset, you can run the following command

```shell
python tools/misc/browse_dataset.py configs/_base_/datasets/kitti-3d-3class.py --task det --output-dir ${OUTPUT_DIR} --online
```

**Notice**: Once specifying `--output-dir`, the images of views specified by users will be saved when pressing `_ESC_` in open3d window. If you don't have a monitor, you can remove the `--online` flag to only save the visualization results and browse them offline.

To verify the data consistency and the effect of data augmentation, you can also add `--aug` flag to visualize the data after data augmentation using the command as below:

```shell
python tools/misc/browse_dataset.py configs/_base_/datasets/kitti-3d-3class.py --task det --aug --output-dir ${OUTPUT_DIR} --online
```

If you also want to show 2D images with 3D bounding boxes projected onto them, you need to find a config that supports multi-modality data loading, and then change the `--task` args to `multi_modality-det`. An example is showed below

```shell
python tools/misc/browse_dataset.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py --task multi_modality-det --output-dir ${OUTPUT_DIR} --online
```

![](../../resources/browse_dataset_multi_modality.png)

You can simply browse different datasets using different configs, e.g. visualizing the ScanNet dataset in 3D semantic segmentation task

```shell
python tools/misc/browse_dataset.py configs/_base_/datasets/scannet_seg-3d-20class.py --task seg --output-dir ${OUTPUT_DIR} --online
```

![](../../resources/browse_dataset_seg.png)

And browsing the nuScenes dataset in monocular 3D detection task

```shell
python tools/misc/browse_dataset.py configs/_base_/datasets/nus-mono3d.py --task mono-det --output-dir ${OUTPUT_DIR} --online
```

![](../../resources/browse_dataset_mono.png)
