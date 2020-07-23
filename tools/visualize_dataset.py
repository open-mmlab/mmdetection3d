import numpy as np
import mmcv, torch, os, argparse

from mmdet3d.apis.inference import *
from mmdet3d.datasets import build_dataset


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

    parser.add_argument("cfg", type=str,
        default="configs/subset/dataset/dataset.py",
        help="Config file")

    parser.add_argument("--idx", type=int, nargs='+',
        default=0, help="Sample index in the dataset")

    parser.add_argument("--outdir", type=str,
        default='cache/', help="Path to the output directory")

    args = parser.parse_args()

    # Build dataset
    cfg = mmcv.Config.fromfile(args.cfg)
    dataset = build_dataset(cfg.data.train)

    img_norm_cfg = cfg.get('img_norm_cfg', None)
    if img_norm_cfg is not None:
        _MEAN = torch.tensor(img_norm_cfg['mean'])
        _STD = torch.tensor(img_norm_cfg['std'])

    # Execution
    def execution(idx):

        # Select and infer a sample
        print("\nProcessing idx {}...".format(idx))
        sample = dataset.__getitem__(idx)

        # Get img_file
        img_file = None
        if isinstance(sample['img_metas'], list):
            if 'filename' in sample['img_metas'][0].data:
                img_file = sample['img_metas'][0].data['filename']
        else:
            if 'filename' in sample['img_metas'].data:
                img_file = sample['img_metas'].data['filename']
        print("img_file:", img_file)

        # Decode PointPainting if having
        points = sample['points'].data
        print("points.shape:", points.shape)

        if (points.shape[1] == 7) and img_file is not None:  # RGB painting
            points[:, 4:] = torch.clamp(points[:, 4:] * _STD + _MEAN, 0, 255)

        # Convert to MeshLab
        data = dict(
            img_metas=[[sample['img_metas'].data]],
            points=[[points]],
        )
        gts = dict(
            boxes_3d=sample['gt_bboxes_3d'].data,
            scores_3d=torch.ones([len(sample['gt_labels_3d'].data)]),
            labels_3d=sample['gt_labels_3d'].data,
        )
        print("Number of objects:", len(sample['gt_labels_3d'].data))

        os.makedirs(args.outdir, exist_ok=True)
        show_result_meshlab(data, gts, args.outdir)
        print("Visualization result is saved in \'{}\'."
            "Please use MeshLab to visualize it.".format(args.outdir))

    [execution(idx) for idx in args.idx]
