# author: Xinge, Tai
# @file: shape_sig_lyft.py
import argparse
import mmcv
import numpy as np
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import points_in_box, view_points
from numpy.polynomial import chebyshev as chebyshev
from os import path as osp
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull


def get_points(sample_id):
    """Get point cloud data given the sample index.

    Args:
        sample_id (str): Sample index.

    Returns:
        np.ndarray: Point cloud data of corresponding to the given sample id.
    """
    sample = level5data.sample[sample_id]
    lidar_path, boxes, _ = \
        level5data.get_sample_data(sample['data']['LIDAR_TOP'])

    points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1)

    if np.remainder(len(points), 5) != 0:
        cnt = len(points) - np.remainder(len(points), 5)
        print('Abnormal data:')
        print(str(lidar_path))
        points = points[:cnt].reshape([-1, 5])
    else:
        points = points.reshape([-1, 5])

    points[:, 3] /= 255
    points[:, 4] = 0

    return points, boxes


def get_points_after_view_transform(points, box_t):
    """Get three views of points in the bounding box.

    Args:
        points (np.ndarray): Point cloud data of the whole scene.
        box_t (:obj:`Box`): The input ground truth bounding box.

    Returns:
        np.ndarray: Point cloud data of corresponding to the given sample id.
    """
    # 1. get the points in the bounding box.
    points_new = points[:, 0:3].transpose()
    mask = points_in_box(box_t, points_new, wlh_factor=1.0)
    if mask.sum() <= 1:
        global number_of_less
        number_of_less += 1
        return None
    points_in = points_new.transpose()[mask]
    # normalize the 3D points
    points_in = points_in - box_t.center
    points_in = points_in.transpose()

    # 2. get the normal vector of forwarding direction
    box_coors = box_t.corners().transpose()
    pq = box_coors[3] - box_coors[0]
    pr = box_coors[4] - box_coors[0]
    normal_1 = np.cross(pq, pr)

    # 3. get the rotation angle between normal vector and base vector
    def unit_vector(vector):
        """Returns the unit vector of the vector.

        Args:
            vector (np.ndarray): The input vector.

        Returns:
            np.ndarray: The unit vector of the input.
        """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """Returns the angle in radians between vectors v1 and v2.

        Examples:

        ..code-block::

        >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793

        Args:
            v1 (np.ndarray): Input vector v1.
            v2 (np.ndarray): Input vector v2.

        Returns:
            np.float64: The angle in radians between v1 and v2.
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    base_vector = np.array([0, 1, 0])
    angle_t = angle_between(normal_1, base_vector)

    # get the axis of rotation
    axis_t = np.cross(normal_1, base_vector)

    # 4. get the matrix representation of view
    # angle --> Quaternion --> matrix representation
    quat = Quaternion._from_axis_angle(axis_t, angle_t)
    FLOAT_EPS = np.finfo(np.float).eps

    def _quat_to_matrix(quat):
        """Quaternion to matrix.

        Args:
            quat (:obj:`Quaternion`): Input quaternion.

        Returns:
            np.ndarray: Matrix representation of the input quaternion.
        """
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        Nq = w * w + x * x + y * y + z * z
        if Nq < FLOAT_EPS:
            return np.eye(3)
        s = 2.0 / Nq
        X = x * s
        Y = y * s
        Z = z * s
        wX = w * X
        wY = w * Y
        wZ = w * Z
        xX = x * X
        xY = x * Y
        xZ = x * Z
        yY = y * Y
        yZ = y * Z
        zZ = z * Z
        return np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY],
                         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])

    view_t = _quat_to_matrix(quat)

    # 5. view transformation
    points_t = view_points(points_in, view_t, normalize=False)
    return points_t


def get_hull_from_three_views(points_t):
    """Compute the symmetry  and convex hull.

    Args:
        points_t (np.ndarray): Transformed points data.

    Returns:
        tuple[:obj:`ConvexHull`]: Convex Hull of bird view/front view and \
            side view.
    """
    points_trans = points_t.transpose()
    bird_view = np.concatenate(
        (points_trans[:, [0, 1]], points_trans[:, [0, 1]] * [-1, -1]), 0)
    front_view = np.concatenate(
        (points_trans[:, [1, 2]], points_trans[:, [1, 2]] * [-1, -1]), 0)
    profile_view = np.concatenate(
        (points_trans[:, [0, 2]], points_trans[:, [0, 2]] * [-1, -1]), 0)

    hull_bird = ConvexHull(bird_view)
    hull_forw = ConvexHull(front_view)
    hull_prof = ConvexHull(profile_view)

    return hull_bird, hull_forw, hull_prof


def hit(U, hull):
    """Compute the distance between line and polygan.

    Args:
        U (np.ndarray): Input line vector.
        hull (:obj:`ConvexHull`): Input convex hull.

    Returns:
        np.ndarray: The distance between line and polygan.
    """
    eq = hull.equations.T
    V, b = eq[:-1], eq[-1]
    alpha = -b / np.dot(V.transpose(), U)
    return np.min(alpha[alpha > 0]) * U


def compute_dist(degs, hull):
    """Compute the rotated distance.

    Args:
        degs (list): Input degree list.
        hull (:obj:`ConvexHull`): Input convex hull.

    Returns:
        list[np.ndarray]: Distance list corresponding to input degrees.
    """
    dist_p_hull_list = []
    for deg in degs:
        if deg < 90:
            y_ = np.tan(np.radians(deg))
            dist_p_hull = hit(np.array([1.0, y_]), hull)
        elif deg == 90:
            dist_p_hull = hit(np.array([0, 1.0]), hull)
        elif deg > 90 and deg < 180:
            y_ = -1.0 * np.tan(np.radians(deg))
            dist_p_hull = hit(np.array([-1, y_]), hull)
        elif deg == 180:
            dist_p_hull = hit(np.array([-1.0, 0.0]), hull)
        elif deg > 180 and deg < 270:
            y_ = -1.0 * np.tan(np.radians(deg))
            dist_p_hull = hit(np.array([-1, y_]), hull)
        elif deg == 270:
            dist_p_hull = hit(np.array([0.0, -1.0]), hull)
        elif deg > 270 and deg < 360:
            y_ = np.tan(np.radians(deg))
            dist_p_hull = hit(np.array([1.0, y_]), hull)
        elif deg == 0 or deg == 360:
            dist_p_hull = hit(np.array([1.0, 0.0]), hull)
        else:
            print('deg error !!!')
        dist_p_hull_list.append([dist_p_hull[0], dist_p_hull[1]])

    return dist_p_hull_list


def get_coff(hull, degs):
    """Compute coefficient given the hull and degrees.

    Args:
        hull (:obj:`ConvexHull`): Input convex hull.
        degs (list): Input degrees list.

    Returns:
        np.ndarray: Computed Chebyshev coefficients.
    """
    dist_degs = np.array(compute_dist(degs, hull))
    dist_len = np.sqrt(dist_degs[:, 0]**2 + dist_degs[:, 1]**2)
    coefficient, _ = chebyshev.chebfit(degs, dist_len, 8, full=True)
    return coefficient


def get_coff_three_views(sample_id, shape_emb):
    """Compute coefficient of three views.

    Args:
        sample_id (str): Input sample index.
        shape_emb (dict[np.ndarray]): Collections of all shape embeddings.
    """
    global shape_pattern

    points, boxes = get_points(sample_id)
    print('len of boxes: ', len(boxes))
    for box_id in range(len(boxes)):
        box_t = boxes[box_id]
        token_t = box_t.token
        if box_t.name not in shape_pattern.keys():
            continue
        name_t = box_t.name
        xyz = box_t.center
        dist_t = np.sqrt(xyz[0]**2 + xyz[1]**2)
        point_trans = get_points_after_view_transform(points, box_t)

        item_t = {'name': name_t, 'dist': dist_t}

        if token_t in shape_emb.keys():
            print('Not Unique Token: ', token_t)

        if point_trans is None:
            item_t['shape'] = None
            shape_emb[token_t] = item_t
            continue
        bird_hull, front_hull, prof_hull = \
            get_hull_from_three_views(point_trans)
        degs = np.arange(360) * (360 / 360)
        coff_bird = get_coff(bird_hull, degs)
        coff_forw = get_coff(front_hull, degs)
        coff_prof = get_coff(prof_hull, degs)

        coff = np.concatenate((coff_bird[0:3], coff_forw[0:3], coff_prof[0:3]),
                              0)
        item_t['shape'] = coff
        shape_pattern[name_t].append(list(coff))
        shape_emb[token_t] = item_t


def get_all_shape_embedding():
    """Compute all shape embeddings."""
    shape_emb = {}
    for sample_id in range(len(level5data.sample)):
        get_coff_three_views(sample_id, shape_emb)
    print('total boxes: ', len(shape_emb.keys()))

    mmcv.dump(shape_emb, osp.join(args.root_path, 'box_shape.pkl'))


parser = argparse.ArgumentParser(
    description='Shape signature generator arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/lyft/',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.01',
    required=False,
    help='specify the dataset version')
args = parser.parse_args()

if __name__ == '__main__':
    level5data = LyftDataset(
        data_path=osp.join(args.root_path, f'{args.version}-train'),
        json_path=osp.join(args.root_path, f'{args.version}-train',
                           f'{args.version}-train'),
        verbose=True)
    number_of_less = 0
    shape_pattern = {
        'bicycle': [],
        'bus': [],
        'car': [],
        'motorcycle': [],
        'pedestrian': [],
        'truck': [],
        'other_vehicle': [],
        'emergency_vehicle': [],
        'animal': []
    }
    get_all_shape_embedding()
    print('less of 2 points: ', number_of_less)

    avg_shape = {}
    for keys_ in shape_pattern.keys():
        shape_com = shape_pattern[keys_]
        if len(shape_com) > 0:
            print('Keys: ', keys_)
            print('Common Values: ',
                  np.array(shape_com).sum(0) / len(shape_com))
            avg_shape[keys_] = \
                np.array(shape_com).sum(0) / len(shape_com)

    print('Fixing the pickle file.\n')
    print('Replace the signature of objects with less than' +
          '2 points by average signature...')

    shape_emb = mmcv.load(osp.join(args.root_path, 'box_shape.pkl'))

    for i, key in enumerate(shape_emb):
        print('shape_emb sample: %d/%d' % (i + 1, len(shape_emb)))
        if shape_emb[key]['shape'] is None:
            shape_emb[key]['shape'] = avg_shape[shape_emb[key]['name']]

    mmcv.dump(shape_emb, osp.join(args.root_path, 'box_shape.pkl'))
