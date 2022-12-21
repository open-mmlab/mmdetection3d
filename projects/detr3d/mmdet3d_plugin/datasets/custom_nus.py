# from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
# from typing import Callable, List, Union
# from mmdet3d.structures.bbox_3d.utils import get_lidar2img
# import torch

# from mmdet3d.registry import DATASETS
# @DATASETS.register_module()
# class custom_nuscenes(NuScenesDataset):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
#         data_info = super().parse_data_info(info)
#         for cam in data_info['images']:
#             lidar2cam = torch.Tensor(data_info['images'][cam]['lidar2cam'])
#             cam2img   = torch.Tensor(data_info['images'][cam]['cam2img'])
#             lidar2img = get_lidar2img(cam2img, lidar2cam)
#             data_info['images'][cam]['lidar2img'] = lidar2img.numpy()
#         return data_info
