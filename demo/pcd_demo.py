import os
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_detector, show_result_meshlab


def main():
    parser = ArgumentParser()
    parser.add_argument('demo_object', help='single pcd file or dataset dir')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-mini',
        help='specify the dataset version, no need for kitti')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.6, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single pcd file or dataset
    if os.path.isfile(args.demo_object):
        result, data = inference_detector(model, pcd=args.demo_object)
        show_result_meshlab(data, result, args.out_dir)
    elif os.path.isdir(args.demo_object):
        result, data = inference_detector(
            model, data_root=args.demo_object, version=args.version)
        show_result_meshlab(data, result, args.out_dir)
    else:
        print('Error!!! Please input the correct file or dir')
        exit(0)
    # show the results
    show_result_meshlab(data, result, args.out_dir)


if __name__ == '__main__':
    main()
