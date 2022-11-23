# Data Information Protocol

In MMDetection3D 1.1, we unify the modalities, tasks of dataset in one `.pkl` file, which stores a dict containing two keys: `metainfo` and `data_list`.

`metainfo` contains the basic information for the dataset itself, such as `categories`, `dataset` and `info_version`, while `data_list` is a list of dict, each dict (hereinafter referred to as `info`) contains all the detailed information of single sample as follows:

- info\['sample_idx'\]: The index of this sample in the whole dataset.
- info\['images'\]: Information of images captured by multiple cameras. A dict contains six keys corresponding to each camera: `'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_BACK'`, `'CAM_BACK_LEFT'`, `'CAM_BACK_RIGHT'`. Each dict contains all data information related to corresponding camera.
- info\['images'\]\['CAM_XXX'\]: Include some information about the `CAM2` camera sensor.
  - info\['images'\]\['CAM_XXX'\]\['img_path'\]: The filename of the image.
  - info\['images'\]\['CAM_XXX'\]\['height'\]: The height of the image.
  - info\['images'\]\['CAM_XXX'\]\['width'\]: The width of the image.
  - info\['images'\]\['CAM_XXX'\]\['cam2img'\]: Transformation matrix from camera to image with shape (4, 4).
  - info\['images'\]\['CAM_XXX'\]\['lidar2cam'\]: Transformation matrix from lidar to camera with shape (4, 4).
- info\['lidar_points'\]: A dict containing all the information related to the lidar points.
  - info\['lidar_points'\]\['lidar_path'\]: The filename of the lidar point cloud data.
  - info\['lidar_points'\]\['num_pts_feats'\]: The feature dimension of point.
- info\['radar_points'\]: A dict containing all the information related to the lidar points.
  - info\['radar_points'\]\['radar_path'\]: The filename of the radar point cloud data.
  - info\['radar_points'\]\['num_pts_feats'\]: The feature dimension of point.
- info\['lidar_sweeps'\]: A list contains sweeps information (The intermediate lidar frames without annotations)
  - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['data_path'\]: The lidar data path of i-th sweep.
  - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['lidar2ego'\]: The transformation matrix from this lidar sensor to ego vehicle. (4x4 list)
  - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['ego2global'\]: The transformation matrix from the ego vehicle to global coordinates. (4x4 list)
  - info\['lidar_sweeps'\]\[i\]\['lidar2sensor'\]: The transformation matrix from the main lidar sensor to the current sensor (for collecting the sweep data). (4x4 list)
  - info\['lidar_sweeps'\]\[i\]\['timestamp'\]: Timestamp of the sweep data.
  - info\['lidar_sweeps'\]\[i\]\['sample_data_token'\]: The sweep sample data token.
- info\['instances'\]: It is a list of dict. Each dict contains all annotation information of single instance. For the i-th instance:
  - info\['instances'\]\[i\]\['bbox_3d'\]: List of 7 numbers representing the 3D bounding box of the instance, in (x, y, z, w, h, l, yaw) order.
  - info\['instances'\]\[i\]\['bbox_label_3d'\]: An int indicate the 3D label of instance and the -1 indicating ignore.
- info\['cam_instances'\]: It is a dict contains keys `'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_BACK'`, `'CAM_BACK_LEFT'`, `'CAM_BACK_RIGHT'`. For vision-based 3D object detection task, we split 3D annotations of the whole scenes according to the camera they belong to.
  - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_label'\]: Label of instance.
  - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_label_3d'\]: Label of instance.
  - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox'\]: 2D bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as \[x1, y1, x2, y2\].
  - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['center_2d'\]: Projected center location on the image, a list has shape (2,), .
  - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['depth'\]: The depth of projected center.
  - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_3d'\]: List of 7 numbers representing the 3D bounding box of the instance, in (x, y, z, l, h, w, yaw) order.
- info\['plane'\](optional): Road level information.

We unify the data info storage protocol in such following format:

01. sample_idx(str): Sample index of the frame.
02. token (str, optional): '000010'
03. timestamp (float, optional) : Timestamp of the current frame.
04. ego2global (list\[list\[float\]\], optional) : Transformation matrix from ego-vehicle to the global coordinate system with shape \[4, 4\]. (pose in KITTI and Waymo)
05. images (dict\[dict\], optional): Information of images captured by multiple cameras, with keys like CAM_FRONT.
06. img_path(str, required): the full path to the image file.
07. height, width(int, required):The shape of the image.
08. depth_map (str, optional): Path of the depth map file.
09. cam2img (list\[list\[float\]\], optional) : Transformation matrix from camera to image with shape \[3, 3\], \[3, 4\] or \[4, 4\].
10. sensor2ego /cam2ego (list\[list\[float\]\], optional) : Transformation matrix from camera to ego-vehicle with shape \[4, 4\].
11. lidar_points (dict, optional) Each dict contains information of LiDAR point cloud frame.
12. num_pts_feats (int, optional) : Number of features for each point.
13. lidar_path (str, optional): Path of LiDAR data file.
14. sensor2ego /lidar2ego (list\[list\[float\]\], optional): Transformation matrix from lidar to ego-vehicle with shape \[4, 4\]. (Referenced camera coordinate system is ego in KITTI.)
15. lidar2img/depth2img (list\[list\[float\]\], optional): Transformation matrix from lidar or depth to image with shape \[4, 4\]. (待定)
16. radar_points (dict, optional) Each dict contains information of Radar point cloud frame.
17. num_pts_feats (int, optional) : Number of features for each point.
18. radar_path (str, optional): Path of Radar data file.
19. sensor2ego /radar2ego (list\[list\[float\]\], optional): Transformation matrix from radar to ego-vehicle with shape \[4, 4\]. (Referenced camera coordinate system is ego in KITTI.)
20. image_sweeps(list\[dict\], optional): Image sweeps data.
21. ego2global (list\[list\[float\]\], optional) : Transformation matrix from ego-vehicle to global with shape \[4, 4\].
22. timestamp (float, optional) : Timestamp of the sweep.
23. images (dict\[dict\]): Information of images captured by multiple cameras, with keys like CAM_FRONT.
    1\. img_path(str, required): the full path to the image file.
    2\. depth_map (str, optional): Path of the depth map file.
    3\. height, width(int, optional):The shape of the image.
    4\. cam2img (list\[list\[float\]\], optional) : Transformation matrix from camera to image with shape \[3, 3\], \[3, 4\] or \[4, 4\].
    5\. sensor2ego /cam2ego (list\[list\[float\]\], optional) : Transformation matrix from camera to ego-vehicle with shape \[4, 4\].
24. lidar_sweeps(list\[dict\], optional): LiDAR points sweeps data.
25. ego2global (list\[list\[float\]\], optional) : Transformation matrix from ego-vehicle to global with shape \[4, 4\].
26. timestamp (float, optional) : Timestamp of the sweep.
27. lidar_points (dict, optional) Each dict contains information of LiDAR point cloud frame.
    1\. num_pts_feats (int, optional) : Number of features for each point.
    2\. lidar_path (str, optional): Path of LiDAR data file.
    3\. sensor2ego /lidar2ego (list\[list\[float\]\], optional): Transformation matrix from lidar to ego-vehicle with shape \[4, 4\]. (Referenced camera coordinate system is ego in KITTI.)
    4\. lidar2img/depth2img (list\[list\[float\]\], optional): Transformation matrix from lidar or depth to image with shape \[4, 4\]. (待定)
28. instances (list\[dict\], optional): Required by object detection, instance detection/segmentation or keypoint detection tasks. Each dict corresponds to annotations of one instance in this image, and may contain the following keys:
29. bbox (list\[float\], required): list of 4 numbers representing the 2D bounding box of the instance, in (x1, y1, x2, y2) order (exterior rectangle of the projected 3D box).
30. bbox_label (int, required): an integer in the range \[0, num_categories-1\] representing the category label.
31. tight_bbox (list\[float\], optional): list of 4 numbers representing the manually annotated 2D bounding box of the instance in (x1, y1, x2, y2)  order. This label currently occurs in KITTI and Waymo dataset.
32. bbox_3d (list\[float\], optional): list of 7 (or 9) numbers representing the 3D bounding box of the instance, in \[x, y, z, w, h, l, yaw\] (or \[x, y, z, w, h, l, yaw, vx, vy\]) order.
33. bbox_3d_isvalid (bool, optional): Whether to use the 3D bounding box during training.
34. bbox_label_3d(int, optional): 3D category label (typically the same as label).
35. depth (float, optional): Projected center depth of the 3D bounding box compared to the image plane. (待定)
36. center_2d (list\[float\], optional): Projected 2D center of the 3D bounding box. (待定)
37. attr_label (int, optional): Attribute labels (fine-grained labels such as stopping, moving, ignore, crowd).
38. num_lidar_pts (int, optional): The number of LiDAR points in the 3D bounding box.
39. num_radar_pts (int, optional): The number of Radar points in the 3D bounding box.
40. difficulty (int, optional): Difficulty level of detecting the 3D bounding box.
41. unaligned_bbox_3d(待定)
42. instances_ignore (list\[dict\], optional): Required by object detection, instance  to be ignored during training. Each dict corresponds to annotations of one instance in this image, and may contain the following keys:
43. pts_semantic_mask_path (str, optional): Path of semantic labels for each point.
44. pts_instance_mask_path (str, optional): Path of instance labels for each point.
