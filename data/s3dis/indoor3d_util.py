import glob
import numpy as np
import os

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

g_classes = [
    x.rstrip()
    for x in open(os.path.join(BASE_DIR, 'meta_data/class_names.txt'))
]
g_class2label = {cls: i for i, cls in enumerate(g_classes)}
g_class2color = {
    'ceiling': [0, 255, 0],
    'floor': [0, 0, 255],
    'wall': [0, 255, 255],
    'beam': [255, 255, 0],
    'column': [255, 0, 255],
    'window': [100, 100, 255],
    'door': [200, 200, 100],
    'table': [170, 120, 200],
    'chair': [255, 0, 0],
    'sofa': [200, 100, 100],
    'bookcase': [10, 200, 100],
    'board': [200, 200, 200],
    'clutter': [50, 50, 50]
}
g_easy_view_labels = [7, 8, 9, 10, 11, 1]
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}

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

    Returns:
        None

    Note:
        the points are shifted before save, the most negative point is now
            at origin.
    """
    points_list = []
    ins_idx = 1  # instance ids should be indexed from 1, so 0 is unannotated

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in g_classes:  # note: in some room there is 'staris' class
            cls = 'clutter'
        points = np.loadtxt(f)
        labels = np.ones((points.shape[0], 1)) * g_class2label[cls]
        ins_labels = np.ones((points.shape[0], 1)) * ins_idx
        ins_idx += 1
        points_list.append(np.concatenate([points, labels, ins_labels], 1))

    data_label = np.concatenate(points_list, 0)  # [N, 8], (pts, rgb, sem, ins)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min

    np.save(f'{out_filename}_point.npy', data_label[:, :6].astype(np.float32))
    np.save(f'{out_filename}_sem_label.npy', data_label[:, 6].astype(np.int))
    np.save(f'{out_filename}_ins_label.npy', data_label[:, 7].astype(np.int))
