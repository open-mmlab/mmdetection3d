# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
from ts.torch_handler.base_handler import BaseHandler

from mmdet3d.apis import inference_detector, init_model


class MMdet3dHandler(BaseHandler):
    threshold = 0.5

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')
        self.model = init_model(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data):

        return data

    def inference(self, model_input):
        results, data = inference_detector(self.model, model_input)
        return results

    def postprocess(self, inference_output):
        # Format output following the example ObjectDetectionHandler format
        output = []
        result = inference_output[0]
        if 'pts_bbox' in result[0].keys():
            pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
            pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
        else:
            pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
            pred_scores = result[0]['scores_3d'].numpy()

        for index in range(len(pred_bboxes)):
            bbox_coords = pred_bboxes[index]
            score = pred_scores[index]

            if score > self.threshold:
                output.append({'3dbbox': bbox_coords, 'score': score})

        return output
