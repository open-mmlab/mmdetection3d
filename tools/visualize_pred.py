import numpy as np
from copy import deepcopy
import mmcv, torch, os, argparse

from mmdet3d.apis.inference import *
from mmdet3d.apis import init_detector
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines import Compose


#------------------------------------------------------------------------------
#  Utils
#------------------------------------------------------------------------------
def show_result_meshlab(data, result, out_dir):
    points = data['points'][0][0].cpu().numpy()
    pts_filename = data['img_metas'][0][0]['pts_filename']
    file_name = osp.split(pts_filename)[-1].split('.')[0]

    assert out_dir is not None, 'Expect out_dir, got none.'

    pred_bboxes = result['boxes_3d'].tensor.numpy()
    # for now we convert points into depth mode
    if data['img_metas'][0][0]['box_mode_3d'] != Box3DMode.DEPTH:
        channel_ids = [1, 0, 2]
        if points.shape[1] > 4:
            channel_ids += list(range(points.shape[1]))[4:]

        points = points[..., channel_ids]
        points[..., 0] *= -1
        pred_bboxes = Box3DMode.convert(pred_bboxes,
                                        data['img_metas'][0][0]['box_mode_3d'],
                                        Box3DMode.DEPTH)
        pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2
    else:
        pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2
    show_result(points, None, pred_bboxes, out_dir, file_name)


def inference_detector(model, data):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        raise NotImplementedError('Not support cpu-only currently')

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


#------------------------------------------------------------------------------
#  Main execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument("cfg", type=str,
        default="configs/subset/hv_pointpillars_secfpn_6x2_80e_kitti-3d-car.py",
        help="Config file")

    parser.add_argument("--ckpt", type=str,
        default="../../checkpoints/3d/hv_pointpillars_secfpn_6x2_80e_kitti-3d-car/epoch_40.pth",
        help="Checkpoint file")

    parser.add_argument("--idx", type=int, nargs='+',
        default=0, help="Sample index in the dataset")

    parser.add_argument("--outdir", type=str,
    default='cache/', help="Path to the output directory")

    args = parser.parse_args()

    # Build dataset and model
    cfg = mmcv.Config.fromfile(args.cfg)
    dataset = build_dataset(cfg.data.test)

    img_norm_cfg = cfg.get('img_norm_cfg', None)
    if img_norm_cfg is not None:
        _MEAN = torch.tensor(img_norm_cfg['mean'])
        _STD = torch.tensor(img_norm_cfg['std'])

    model = init_detector(args.cfg, args.ckpt, device='cuda')

    # Execution function
    def execution(idx):
        global dataset, model, cfg, args
        print()

        # Select and infer a sample
        data = dataset.__getitem__(idx)
        if isinstance(data['img_metas'], list):
            pcd_filename = data['img_metas'][0].data['pts_filename']
            img_file = data['img_metas'][0].data['filename']
        else:
            pcd_filename = data['img_metas'].data['pts_filename']
            img_file = data['img_metas'].data['filename']

        pcd_file = os.path.join(dataset.root_split, dataset.pts_prefix, pcd_filename)
        print("pcd_file:", pcd_file)
        print("img_file:", img_file)

        result, data = inference_detector(model, data)
        print("Number of detected bboxes:", len(result['labels_3d']))

        # Decode PointPainting if having
        points = data['points'][0][0].data.cpu()
        if points.shape[1] == 7:  # RGB painting
            points[:, 4:] = torch.clamp(points[:, 4:] * _STD + _MEAN, 0, 255)
        data['points'][0][0].data = points.cuda()

        # Convert to MeshLab
        os.makedirs(args.outdir, exist_ok=True)
        show_result_meshlab(data, result, args.outdir)
        print("Visualization result is saved in \'{}\'. "
            "Please use MeshLab to visualize it.".format(
                os.path.join(args.outdir, os.path.basename(pcd_file).split('.')[0])))

    [execution(idx) for idx in args.idx]
