"""Convert PandaSet to MMDetection3D custom dataset format.

Generates MMDetection3D v2 info pickle files for PandaSet. Processes both
sensors in a single pass per frame to minimize I/O (reads each .pkl only once).

Output:
  - data/pandaset/pandaset_pandar64_infos_test.pkl
  - data/pandaset/pandaset_pandargt_infos_test.pkl

Usage:
    python tools/dataset_converters/pandaset_converter.py \
        --pandaset-root data/pandaset/sequences \
        --out-dir data/pandaset \
        --workers 8
"""

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

NUSCENES_CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier')

LABEL_MAP = {
    'Car': 'car',
    'Pickup Truck': 'car',
    'Medium-sized Truck': 'truck',
    'Semi-truck': 'truck',
    'Towed Object': 'trailer',
    'Bus': 'bus',
    'Tram / Subway': 'bus',
    'Other Vehicle - Construction Vehicle': 'construction_vehicle',
    'Bicycle': 'bicycle',
    'Motorcycle': 'motorcycle',
    'Motorized Scooter': 'motorcycle',
    'Pedestrian': 'pedestrian',
    'Pedestrian with Object': 'pedestrian',
    'Cones': 'traffic_cone',
    'Pylons': 'traffic_cone',
    'Road Barriers': 'barrier',
    'Temporary Construction Barriers': 'barrier',
}


def pose_to_matrix(pose_dict):
    """Convert PandaSet pose dict to 4x4 homogeneous matrix."""
    pos = pose_dict['position']
    heading = pose_dict['heading']
    r = Rotation.from_quat(
        [heading['x'], heading['y'], heading['z'], heading['w']])
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = [pos['x'], pos['y'], pos['z']]
    return mat


def transform_box_to_ego(position, yaw, inv_pose, pose_yaw):
    """Transform box center + yaw from world to ego frame."""
    pos_h = np.array([position[0], position[1], position[2], 1.0])
    ego_pos = (inv_pose @ pos_h)[:3]
    ego_yaw = yaw - pose_yaw
    return ego_pos, ego_yaw


def process_sequence(seq_id, pandaset_root):
    """Process all frames of one sequence.

    Returns (list_64, list_gt).
    """
    seq_path = os.path.join(pandaset_root, seq_id)
    poses_file = os.path.join(seq_path, 'lidar', 'poses.json')

    if not os.path.exists(poses_file):
        return [], []

    with open(poses_file) as f:
        poses = json.load(f)

    lidar_dir = os.path.join(seq_path, 'lidar')
    cuboid_dir = os.path.join(seq_path, 'annotations', 'cuboids')

    frame_files = sorted(
        [f for f in os.listdir(lidar_dir) if f.endswith('.pkl')])

    results_64 = []
    results_gt = []

    for fname in frame_files:
        frame_idx = int(fname.replace('.pkl', ''))
        lidar_file = os.path.join(lidar_dir, fname)
        cuboid_file = os.path.join(cuboid_dir, fname)

        if not os.path.exists(cuboid_file):
            continue

        pose_mat = pose_to_matrix(poses[frame_idx])
        inv_pose = np.linalg.inv(pose_mat)
        pose_yaw = Rotation.from_matrix(pose_mat[:3, :3]).as_euler('xyz')[2]

        # Read lidar once, get counts for both sensors
        lidar_df = pd.read_pickle(lidar_file)
        counts = lidar_df['d'].value_counts()
        num_pts_64 = int(counts.get(0, 0))
        num_pts_gt = int(counts.get(1, 0))

        # Read cuboids once, process annotations
        cuboid_df = pd.read_pickle(cuboid_file)
        instances = []
        for _, row in cuboid_df.iterrows():
            label = row['label']
            if label not in LABEL_MAP:
                continue

            nus_class = LABEL_MAP[label]
            class_idx = NUSCENES_CLASSES.index(nus_class)

            position = np.array(
                [row['position.x'], row['position.y'], row['position.z']])
            yaw = row['yaw']
            dim_l = row['dimensions.y']
            dim_w = row['dimensions.x']
            dim_h = row['dimensions.z']

            ego_pos, ego_yaw = transform_box_to_ego(position, yaw, inv_pose,
                                                    pose_yaw)

            # Convention alignment:
            # PandaSet: yaw=0 → length along Y.
            # MMDet3D: yaw=0 → length along X.
            ego_yaw = ego_yaw + np.pi / 2
            # PandaSet: z = box center. nuScenes model: z = box bottom.
            ego_pos[2] = ego_pos[2] - dim_h / 2.0

            instances.append({
                'bbox_3d': [
                    float(ego_pos[0]),
                    float(ego_pos[1]),
                    float(ego_pos[2]),
                    float(dim_l),
                    float(dim_w),
                    float(dim_h),
                    float(ego_yaw)
                ],
                'bbox_label_3d':
                class_idx,
                'bbox_3d_isvalid':
                True,
            })

        sample_id = f'{seq_id}_{frame_idx:02d}'
        pose_list = pose_mat.tolist()

        if num_pts_64 > 0:
            results_64.append({
                'lidar_points': {
                    'lidar_path': lidar_file,
                    'num_pts_feats': 4,
                    'pandaset_sensor_id': 0,
                    'pandaset_pose': pose_list,
                },
                'instances': instances,
                'sample_idx': sample_id,
                'num_pts': num_pts_64,
            })

        if num_pts_gt > 0:
            results_gt.append({
                'lidar_points': {
                    'lidar_path': lidar_file,
                    'num_pts_feats': 4,
                    'pandaset_sensor_id': 1,
                    'pandaset_pose': pose_list,
                },
                'instances': instances,
                'sample_idx': sample_id,
                'num_pts': num_pts_gt,
            })

    return results_64, results_gt


def _process_sequence_wrapper(args):
    return process_sequence(*args)


def main():
    parser = argparse.ArgumentParser(description='Generate PandaSet info pkls')
    parser.add_argument('--pandaset-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    pandaset_root = args.pandaset_root
    out_base = args.out_dir
    os.makedirs(out_base, exist_ok=True)

    sequences = sorted([
        d for d in os.listdir(pandaset_root)
        if os.path.isdir(os.path.join(pandaset_root, d))
    ])
    print(f'Found {len(sequences)} sequences in {pandaset_root}')

    all_64 = []
    all_gt = []

    tasks = [(seq_id, pandaset_root) for seq_id in sequences]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_sequence_wrapper, t): t[0]
                for t in tasks
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                r64, rgt = future.result()
                all_64.extend(r64)
                all_gt.extend(rgt)
                if done % 10 == 0:
                    print(f'  Sequences done: {done}/{len(sequences)} '
                          f'(frames so far: {len(all_64)} Pandar64, '
                          f'{len(all_gt)} PandarGT)')
    else:
        for i, t in enumerate(tasks):
            r64, rgt = _process_sequence_wrapper(t)
            all_64.extend(r64)
            all_gt.extend(rgt)
            if (i + 1) % 10 == 0:
                print(f'  Sequences done: {i+1}/{len(sequences)}')

    all_64.sort(key=lambda x: x['sample_idx'])
    all_gt.sort(key=lambda x: x['sample_idx'])

    for data_list, name in [(all_64, 'pandaset_pandar64'),
                            (all_gt, 'pandaset_pandargt')]:
        info_pkl = {
            'metainfo': {
                'CLASSES': NUSCENES_CLASSES,
                'DATASET': name,
                'pandaset_root': pandaset_root,
            },
            'data_list': data_list,
        }
        pkl_path = os.path.join(out_base, f'{name}_infos_test.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(info_pkl, f, protocol=2)

        total_inst = sum(len(d['instances']) for d in data_list)
        print(f'\n{name}:')
        print(f'  File: {pkl_path}')
        print(f'  Frames: {len(data_list)}')
        print(f'  Total instances: {total_inst}')

    print('\nDone! Use LoadPointsFromPandaSet pipeline '
          'transform for inference.')


if __name__ == '__main__':
    main()
