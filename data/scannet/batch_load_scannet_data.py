# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations.

Usage example: python ./batch_load_scannet_data.py
"""
import argparse
import datetime
import numpy as np
import os
from load_scannet_data import export
from os import path as osp

SCANNET_DIR = 'scans'
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])


def export_one_scan(scan_name, output_filename_prefix, max_num_point,
                    label_map_file, scannet_dir):
    mesh_file = osp.join(scannet_dir, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = osp.join(scannet_dir, scan_name,
                        scan_name + '.aggregation.json')
    seg_file = osp.join(scannet_dir, scan_name,
                        scan_name + '_vh_clean_2.0.010000.segs.json')
    # includes axisAlignment info for the train set scans.
    meta_file = osp.join(scannet_dir, scan_name, f'{scan_name}.txt')
    mesh_vertices, semantic_labels, instance_labels, instance_bboxes, \
        instance2semantic = export(mesh_file, agg_file, seg_file,
                                   meta_file, label_map_file, None)

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask, :]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    num_instances = len(np.unique(instance_labels))
    print(f'Num of instances: {num_instances}')

    bbox_mask = np.in1d(instance_bboxes[:, -1], OBJ_CLASS_IDS)
    instance_bboxes = instance_bboxes[bbox_mask, :]
    print(f'Num of care instances: {instance_bboxes.shape[0]}')

    N = mesh_vertices.shape[0]
    if N > max_num_point:
        choices = np.random.choice(N, max_num_point, replace=False)
        mesh_vertices = mesh_vertices[choices, :]
        semantic_labels = semantic_labels[choices]
        instance_labels = instance_labels[choices]

    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)
    np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)
    np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)
    np.save(f'{output_filename_prefix}_bbox.npy', instance_bboxes)


def batch_export(max_num_point, output_folder, train_scan_names_file,
                 label_map_file, scannet_dir):
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.mkdir(output_folder)

    train_scan_names = [line.rstrip() for line in open(train_scan_names_file)]
    for scan_name in train_scan_names:
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = osp.join(output_folder, scan_name)
        if osp.isfile(f'{output_filename_prefix}_vert.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done')
            continue
        try:
            export_one_scan(scan_name, output_filename_prefix, max_num_point,
                            label_map_file, scannet_dir)
        except Exception:
            print(f'Failed export scan: {scan_name}')
        print('-' * 20 + 'done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_num_point',
        default=50000,
        help='The maximum number of the points.')
    parser.add_argument(
        '--output_folder',
        default='./scannet_train_instance_data',
        help='output folder of the result.')
    parser.add_argument(
        '--scannet_dir', default='scans', help='scannet data directory.')
    parser.add_argument(
        '--label_map_file',
        default='meta_data/scannetv2-labels.combined.tsv',
        help='The path of label map file.')
    parser.add_argument(
        '--train_scan_names_file',
        default='meta_data/scannet_train.txt',
        help='The path of the file that stores the scan names.')
    args = parser.parse_args()
    batch_export(args.max_num_point, args.output_folder,
                 args.train_scan_names_file, args.label_map_file,
                 args.scannet_dir)


if __name__ == '__main__':
    main()
