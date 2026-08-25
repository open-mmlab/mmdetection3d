"""Convert PandaSet .pkl point clouds to .bin and .pcd formats.

Converts world-frame point clouds to ego-centric frame and saves as:
  - .bin: raw float32 binary (x, y, z, intensity) per point
  - .pcd: binary PCD format (x, y, z, intensity)

Output structure:
  <out-dir>/cloud_bin/pandar64/{seq}/{frame}.bin
  <out-dir>/cloud_bin/pandargt/{seq}/{frame}.bin
  <out-dir>/cloud_pcd/pandar64/{seq}/{frame}.pcd
  <out-dir>/cloud_pcd/pandargt/{seq}/{frame}.pcd

Usage:
    python tools/convert_pandaset_to_bin_pcd.py \
        --pandaset-root data/pandaset/sequences \
        --out-dir data/pandaset \
        --workers 8
"""

import argparse
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np

SKIP_DIRS = {'pkls', 'cloud_bin', 'cloud_pcd'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PandaSet .pkl point clouds to .bin/.pcd')
    parser.add_argument(
        '--pandaset-root',
        required=True,
        help='Path to pandaset sequences directory')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Output directory (cloud_bin/ and cloud_pcd/ created inside)')
    parser.add_argument(
        '--workers', type=int, default=8, help='Number of parallel workers')
    return parser.parse_args()


def pose_to_matrix(pose):
    """Convert pose dict (position + heading quaternion) to 4x4 matrix."""
    from scipy.spatial.transform import Rotation as R
    q = pose['heading']
    rot = R.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    pos = pose['position']
    mat[:3, 3] = [pos['x'], pos['y'], pos['z']]
    return mat


def write_pcd_binary(filepath, points):
    """Write points (N, 4) as binary PCD file."""
    n = len(points)
    points_f32 = points.astype(np.float32)
    with open(filepath, 'wb') as f:
        header = ('# .PCD v0.7 - Point Cloud Data file format\n'
                  'VERSION 0.7\n'
                  'FIELDS x y z intensity\n'
                  'SIZE 4 4 4 4\n'
                  'TYPE F F F F\n'
                  'COUNT 1 1 1 1\n'
                  f'WIDTH {n}\n'
                  'HEIGHT 1\n'
                  'VIEWPOINT 0 0 0 1 0 0 0\n'
                  f'POINTS {n}\n'
                  'DATA binary\n')
        f.write(header.encode('ascii'))
        f.write(points_f32.tobytes())


def process_sequence(args):
    """Process all frames in a sequence."""
    seq_id, pandaset_root, out_bin, out_pcd = args
    seq_dir = os.path.join(pandaset_root, seq_id)
    lidar_dir = os.path.join(seq_dir, 'lidar')
    poses_file = os.path.join(lidar_dir, 'poses.json')

    if not os.path.isfile(poses_file):
        return f'{seq_id}: SKIP (no poses.json)'

    with open(poses_file, 'r') as f:
        poses = json.load(f)

    frame_files = sorted(
        [f for f in os.listdir(lidar_dir) if f.endswith('.pkl')])
    converted = 0

    for frame_file in frame_files:
        frame_idx = int(frame_file.replace('.pkl', ''))
        frame_path = os.path.join(lidar_dir, frame_file)

        with open(frame_path, 'rb') as f:
            df = pickle.load(f)

        pose = poses[frame_idx]
        pose_mat = pose_to_matrix(pose)
        inv_pose = np.linalg.inv(pose_mat)

        world_xyz = df[['x', 'y', 'z']].values.astype(np.float64)
        intensity = df['i'].values.astype(np.float32)
        device = df['d'].values  # 0=Pandar64, 1=PandarGT

        pts_h = np.hstack([world_xyz, np.ones((len(world_xyz), 1))])
        ego_xyz = (inv_pose @ pts_h.T).T[:, :3].astype(np.float32)

        mask_p64 = (device == 0)
        mask_pgt = (device == 1)
        frame_name = f'{frame_idx:02d}'

        for sensor_name, mask in [('pandar64', mask_p64),
                                  ('pandargt', mask_pgt)]:
            if mask.sum() == 0:
                continue

            pts = np.column_stack([ego_xyz[mask],
                                   intensity[mask]]).astype(np.float32)

            bin_dir = os.path.join(out_bin, sensor_name, seq_id)
            os.makedirs(bin_dir, exist_ok=True)
            pts.tofile(os.path.join(bin_dir, f'{frame_name}.bin'))

            pcd_dir = os.path.join(out_pcd, sensor_name, seq_id)
            os.makedirs(pcd_dir, exist_ok=True)
            write_pcd_binary(os.path.join(pcd_dir, f'{frame_name}.pcd'), pts)

        converted += 1

    return f'{seq_id}: {converted} frames converted'


def main():
    args = parse_args()
    pandaset_root = args.pandaset_root
    out_bin = os.path.join(args.out_dir, 'cloud_bin')
    out_pcd = os.path.join(args.out_dir, 'cloud_pcd')

    all_entries = sorted(os.listdir(pandaset_root))
    sequences = [
        e for e in all_entries if e not in SKIP_DIRS
        and os.path.isdir(os.path.join(pandaset_root, e, 'lidar'))
    ]

    print(f'Found {len(sequences)} sequences')
    print(f'Output .bin: {out_bin}')
    print(f'Output .pcd: {out_pcd}')
    print()

    pool_args = [(s, pandaset_root, out_bin, out_pcd) for s in sequences]
    with Pool(processes=args.workers) as pool:
        results = pool.map(process_sequence, pool_args)

    for r in results:
        print(r)

    total_bin = 0
    for sensor in ['pandar64', 'pandargt']:
        sensor_dir = os.path.join(out_bin, sensor)
        if os.path.exists(sensor_dir):
            for seq in os.listdir(sensor_dir):
                seq_path = os.path.join(sensor_dir, seq)
                if os.path.isdir(seq_path):
                    total_bin += len(os.listdir(seq_path))

    print(f'\nTotal .bin files: {total_bin}')
    print('Done!')


if __name__ == '__main__':
    main()
