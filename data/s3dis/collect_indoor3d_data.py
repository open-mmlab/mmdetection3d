import argparse
from os import path as osp

import mmcv
from indoor3d_util import export

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output-folder',
    default='./s3dis_data',
    help='output folder of the result.')
parser.add_argument(
    '--data-dir',
    default='Stanford3dDataset_v1.2_Aligned_Version',
    help='s3dis data directory.')
parser.add_argument(
    '--ann-file',
    default='meta_data/anno_paths.txt',
    help='The path of the file that stores the annotation names.')
args = parser.parse_args()

anno_paths = [line.rstrip() for line in open(args.ann_file)]
anno_paths = [osp.join(args.data_dir, p) for p in anno_paths]

output_folder = args.output_folder
mmcv.mkdir_or_exist(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6.
# It's fixed manually here.
# Refer to https://github.com/AnTao97/dgcnn.pytorch/blob/843abe82dd731eb51a4b3f70632c2ed3c60560e9/prepare_data/collect_indoor3d_data.py#L18  # noqa
revise_file = osp.join(args.data_dir,
                       'Area_5/hallway_6/Annotations/ceiling_1.txt')
with open(revise_file, 'r') as f:
    data = f.read()
    # replace that extra character with blank space to separate data
    data = data[:5545347] + ' ' + data[5545348:]
with open(revise_file, 'w') as f:
    f.write(data)

for anno_path in anno_paths:
    print(f'Exporting data from annotation file: {anno_path}')
    elements = anno_path.split('/')
    out_filename = \
        elements[-3] + '_' + elements[-2]  # Area_1_hallway_1
    out_filename = osp.join(output_folder, out_filename)
    if osp.isfile(f'{out_filename}_point.npy'):
        print('File already exists. skipping.')
        continue
    export(anno_path, out_filename)
