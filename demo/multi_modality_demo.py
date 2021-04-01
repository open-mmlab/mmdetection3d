from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, inference_multi_modality_detector, init_detector, show_result_meshlab


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('image', help='image file')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result, data = inference_multi_modality_detector(model, args.pcd, args.image, args.ann)
    # show the results
    show_result_meshlab(data, result, args.out_dir)


if __name__ == '__main__':
    main()
