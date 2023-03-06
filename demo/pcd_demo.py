# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()
    return args


def main(args):
    # TODO: Support inference of point cloud numpy file.
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # test a single point cloud sample
    result, data = inference_detector(model, args.pcd)
    points = data['inputs']['points']
    data_input = dict(points=points)

    # show the results
    visualizer.add_datasample(
        'result',
        data_input,
        data_sample=result,
        draw_gt=False,
        show=args.show,
        wait_time=0,
        out_file=args.out_dir,
        pred_score_thr=args.score_thr,
        vis_task='lidar_det')


if __name__ == '__main__':
    args = parse_args()
    main(args)
