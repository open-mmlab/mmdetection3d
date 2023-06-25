import glob
from os import path as osp

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

BASE_DIR = osp.dirname(osp.abspath(__file__))

class_names = [
    x.rstrip() for x in open(osp.join(BASE_DIR, 'meta_data/class_names.txt'))
]
class2label = {one_class: i for i, one_class in enumerate(class_names)}

# -----------------------------------------------------------------------------
# CONVERT ORIGINAL DATA TO POINTS, SEM_LABEL AND INS_LABEL FILES
# -----------------------------------------------------------------------------


def export(anno_path, out_filename):
    """Convert original dataset files to points, instance mask and semantic
    mask files. We aggregated all the points from each instance in the room.

    Args:
        anno_path (str): path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename (str): path to save collected points and labels
        file_format (str): txt or numpy, determines what file format to save.

    Note:
        the points are shifted before save, the most negative point is now
            at origin.
    """
    points_list = []
    ins_idx = 1  # instance ids should be indexed from 1, so 0 is unannotated

    for f in glob.glob(osp.join(anno_path, '*.txt')):
        one_class = osp.basename(f).split('_')[0]
        if one_class not in class_names:  # some rooms have 'staris' class
            one_class = 'clutter'
        points = pd.read_csv(f, header=None, sep=' ').to_numpy()
        labels = np.ones((points.shape[0], 1)) * class2label[one_class]
        ins_labels = np.ones((points.shape[0], 1)) * ins_idx
        ins_idx += 1
        points_list.append(np.concatenate([points, labels, ins_labels], 1))

    data_label = np.concatenate(points_list, 0)  # [N, 8], (pts, rgb, sem, ins)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min

    np.save(f'{out_filename}_point.npy', data_label[:, :6].astype(np.float32))
    np.save(f'{out_filename}_sem_label.npy', data_label[:, 6].astype(np.int64))
    np.save(f'{out_filename}_ins_label.npy', data_label[:, 7].astype(np.int64))
