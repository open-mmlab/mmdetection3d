from pycocotools.coco import COCO

from mmdet3d.core.evaluation.coco_utils import getImgIds
from mmdet.datasets import DATASETS, CocoDataset


@DATASETS.register_module
class NuScenes2DDataset(CocoDataset):

    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def load_annotations(self, ann_file):
        if not self.class_names:
            self.class_names = self.CLASSES
        self.coco = COCO(ann_file)
        # send class_names into the get id
        # in case we only need to train on several classes
        # by default self.class_names = CLASSES
        self.cat_ids = self.coco.getCatIds(catNms=self.class_names)

        self.cat2label = {
            cat_id: i  # + 1 rm +1 here thus the 0-79 are fg, 80 is bg
            for i, cat_id in enumerate(self.cat_ids)
        }
        # send cat ids to the get img id
        # in case we only need to train on several classes
        if len(self.cat_ids) < len(self.CLASSES):
            self.img_ids = getImgIds(self.coco, catIds=self.cat_ids)
        else:
            self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos
