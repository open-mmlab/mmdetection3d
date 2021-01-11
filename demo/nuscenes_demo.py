from argparse import ArgumentParser

from mmdet3d.apis import inference_nuscenes_detector, init_detector, \
     show_nuscenes_result_meshlab

import torch


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='NuScenes Dataset root path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.6, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument("--local_rank", type=int, default=0, help= 'dist parser')

    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result, data = inference_nuscenes_detector(model,args.pcd)
    # show the results
    show_nuscenes_result_meshlab(data, result, args.out_dir)


if __name__ == '__main__':
    main()