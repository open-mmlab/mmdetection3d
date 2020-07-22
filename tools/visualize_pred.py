import numpy as np
import mmcv, torch, os, argparse

from mmdet3d.apis.inference import *
from mmdet3d.datasets import build_dataset
from mmdet3d.apis import init_detector, inference_detector


#------------------------------------------------------------------------------
#  Utils
#------------------------------------------------------------------------------
def show_result_meshlab(data, result, out_dir):
    """Show result by meshlab.

    Args:
        data (dict): Contain data from pipeline.
        result (dict): Predicted result from model.
        out_dir (str): Directory to save visualized result.
    """
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


#------------------------------------------------------------------------------
#  Main execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str,
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
    dataset = build_dataset(cfg.data.val)

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
        sample = dataset.__getitem__(idx)
        if isinstance(sample['img_metas'], list):
            filename = sample['img_metas'][0].data['pts_filename']
        else:
            filename = sample['img_metas'].data['pts_filename']

        pcd_file = os.path.join(dataset.root_split, dataset.pts_prefix, filename)
        print("Selected pcd_file:", pcd_file)

        result, data = inference_detector(model, pcd_file)
        print("Number of detected bboxes:", len(result['labels_3d']))

        # Convert to MeshLab
        os.makedirs(args.outdir, exist_ok=True)
        show_result_meshlab(data, result, args.outdir)
        print("Visualization result is saved in \'{}\'. "
            "Please use MeshLab to visualize it.".format(
                os.path.join(args.outdir, os.path.basename(pcd_file).split('.')[0])))

    [execution(idx) for idx in args.idx]
