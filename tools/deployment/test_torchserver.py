from argparse import ArgumentParser

import numpy as np
import requests

from mmdet3d.apis import inference_detector, init_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='Address and port of the inference server')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='3d bbox score threshold')
    args = parser.parse_args()
    return args


def parse_result(input):
    bbox = input[0]['3dbbox']
    result = np.array(bbox)
    return result


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single point cloud file
    model_result, _ = inference_detector(model, args.pcd)
    # filter the 3d bboxes whose scores > 0.5
    if 'pts_bbox' in model_result[0].keys():
        pred_bboxes = model_result[0]['pts_bbox']['boxes_3d'].numpy()
        pred_scores = model_result[0]['pts_bbox']['scores_3d'].numpy()
    else:
        pred_bboxes = model_result[0]['boxes_3d'].numpy()
        pred_scores = model_result[0]['scores_3d'].numpy()
    model_result = pred_bboxes[pred_scores > 0.5]

    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.pcd, 'rb') as points:
        response = requests.post(url, points)
    server_result = parse_result(response.json())
    assert np.allclose(model_result, server_result)


if __name__ == '__main__':
    args = parse_args()
    main(args)
