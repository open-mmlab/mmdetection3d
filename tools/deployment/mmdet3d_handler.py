# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.structures.points import get_points_type


class MMdet3dHandler(BaseHandler):
    """MMDetection3D Handler used in TorchServe.

    Handler to load models in MMDetection3D, and it will process data to get
    predicted results. For now, it only supports SECOND.
    """
    threshold = 0.5
    load_dim = 4
    use_dim = [0, 1, 2, 3]
    coord_type = 'LIDAR'
    attribute_dims = None

    def initialize(self, context):
        """Initialize function loads the model in MMDetection3D.

        Args:
            context (context): It is a JSON Object containing information
                pertaining to the model artifacts parameters.
        """
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
        """Preprocess function converts data into LiDARPoints class.

        Args:
            data (List): Input data from the request.

        Returns:
            `LiDARPoints` : The preprocess function returns the input
                point cloud data as LiDARPoints class.
        """
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            pts = row.get('data') or row.get('body')
            if isinstance(pts, str):
                pts = base64.b64decode(pts)

            points = np.frombuffer(pts, dtype=np.float32)
            points = points.reshape(-1, self.load_dim)
            points = points[:, self.use_dim]
            points_class = get_points_type(self.coord_type)
            points = points_class(
                points,
                points_dim=points.shape[-1],
                attribute_dims=self.attribute_dims)

        return points

    def inference(self, data):
        """Inference Function.

        This function is used to make a prediction call on the
        given input request.

        Args:
            data (`LiDARPoints`): LiDARPoints class passed to make
                the inference request.

        Returns:
            List(dict) : The predicted result is returned in this function.
        """
        results, _ = inference_detector(self.model, data)
        return results

    def postprocess(self, data):
        """Postprocess function.

        This function makes use of the output from the inference and
        converts it into a torchserve supported response output.

        Args:
            data (List[dict]): The data received from the prediction
                output of the model.

        Returns:
            List: The post process function returns a list of the predicted
                output.
        """
        output = []
        for pts_index, result in enumerate(data):
            output.append([])
            if 'pts_bbox' in result.keys():
                pred_bboxes = result['pts_bbox']['boxes_3d'].tensor.numpy()
                pred_scores = result['pts_bbox']['scores_3d'].numpy()
            else:
                pred_bboxes = result['boxes_3d'].tensor.numpy()
                pred_scores = result['scores_3d'].numpy()

            index = pred_scores > self.threshold
            bbox_coords = pred_bboxes[index].tolist()
            score = pred_scores[index].tolist()

            output[pts_index].append({'3dbbox': bbox_coords, 'score': score})

        return output
