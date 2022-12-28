from glob import glob
from os.path import exists, join

import mmengine
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

tf.enable_eager_execution()


class gt_bin_creator:
    """waymo gt.bin creator.

    Support create gt.bin from tfrecords and gt_subset.bin from gt.bin
    """

    # yapf: disable
    def __init__(
            self,
            ann_file,
            data_root,
            split,  # ('training','validation')
            waymo_bin_file=None,
            load_interval=1,
            for_cam_only_challenge=True,
            file_client_args: dict = dict(backend='disk')):
        # yapf: enable
        self.ann_file = ann_file
        self.waymo_bin_file = waymo_bin_file
        self.data_root = data_root
        self.split = split
        self.load_interval = load_interval
        self.for_cam_only_challenge = for_cam_only_challenge
        self.file_client_args = file_client_args
        self.waymo_tfrecords_dir = join(self.data_root, self.split)
        if self.waymo_bin_file is None:
            self.waymo_bin_file = join(self.data_root,
                                       'gt_{}.bin'.format(self.split))

    def get_target_timestamp(self):
        data_infos = mmengine.load(
            self.ann_file)['data_list'][::self.load_interval]
        self.timestamp = set()
        for info in data_infos:
            self.timestamp.add(info['timestamp'])

    def create_subset(self):
        self.create_whole()

        subset_path = self.waymo_bin_file.replace(
            '.bin', f'_subset_{self.load_interval}.bin')
        if exists(subset_path):
            print(f'file {subset_path} exists. Skipping create_subset')
        else:
            print(f'Can not find {subset_path}, creating a new one...')
            objs = metrics_pb2.Objects()
            objs.ParseFromString(open(self.waymo_bin_file, 'rb').read())
            self.get_target_timestamp()
            objs_subset = metrics_pb2.Objects()
            prog_bar = mmengine.ProgressBar(len(objs.objects))
            for obj in objs.objects:
                prog_bar.update()
                if obj.frame_timestamp_micros not in self.timestamp:
                    continue
                if self.for_cam_only_challenge and \
                   obj.object.type == label_pb2.Label.TYPE_SIGN:
                    continue

                objs_subset.objects.append(obj)

            open(subset_path, 'wb').write(objs_subset.SerializeToString())
            print(f'save subset bin file to {subset_path}')

        return subset_path

    def create_whole(self):
        if exists(self.waymo_bin_file):
            print(f'file {self.waymo_bin_file} exists. Skipping create_whole')
        else:
            print(f'Can not find {self.waymo_bin_file}, creating a new one...')
            self.get_file_names()
            tfnames = self.waymo_tfrecord_pathnames

            objs = metrics_pb2.Objects()
            frame_num = 0
            prog_bar = mmengine.ProgressBar(len(tfnames))
            for i in range(len(tfnames)):
                dataset = tf.data.TFRecordDataset(
                    tfnames[i], compression_type='')
                for data in dataset:
                    frame = open_dataset.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))
                    frame_num += 1
                    for label in frame.laser_labels:
                        if self.for_cam_only_challenge and (
                                label.type == 3
                                or label.camera_synced_box.ByteSize() == 0
                                or label.num_lidar_points_in_box < 1):
                            continue

                        new_obj = metrics_pb2.Object()
                        new_obj.frame_timestamp_micros = frame.timestamp_micros
                        new_obj.object.CopyFrom(label)
                        new_obj.context_name = frame.context.name
                        objs.objects.append(new_obj)
                prog_bar.update()

            open(self.waymo_bin_file, 'wb').write(objs.SerializeToString())
            print(f'Saved groudtruth bin file to {self.waymo_bin_file}\n\
                    It has {len(objs.objects)} objects in {frame_num} frames.')

        return self.waymo_bin_file

    def get_file_names(self):
        """Get file names of waymo raw data."""
        if 'path_mapping' in self.file_client_args:
            for path in self.file_client_args['path_mapping'].keys():
                if path in self.waymo_tfrecords_dir:
                    self.waymo_tfrecords_dir = \
                        self.waymo_tfrecords_dir.replace(
                            path, self.file_client_args['path_mapping'][path])
            from petrel_client.client import Client
            client = Client()
            contents = client.list(self.waymo_tfrecords_dir)
            self.waymo_tfrecord_pathnames = list()
            for content in sorted(list(contents)):
                if content.endswith('tfrecord'):
                    self.waymo_tfrecord_pathnames.append(
                        join(self.waymo_tfrecords_dir, content))
        else:
            self.waymo_tfrecord_pathnames = sorted(
                glob(join(self.waymo_tfrecords_dir, '*.tfrecord')))
        print(len(self.waymo_tfrecord_pathnames), 'tfrecords found.')


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--ann_file',
        default='./data/waymo_dev1x/kitti_format/waymo_infos_val.pkl')
    parser.add_argument(
        '--data_root', default='./data/waymo_dev1x/waymo_format')
    parser.add_argument(
        '--split', default='validation')  # ('training','validation')
    # parser.add_argument('waymo_bin_file')
    parser.add_argument('--load_interval', type=int, default=1)
    parser.add_argument('--for_cam_only_challenge', default=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # test = './data/waymo_dev1x/waymo_format/gt.bin'
    creator = gt_bin_creator(args.ann_file, args.data_root, args.split, None,
                             args.load_interval, args.for_cam_only_challenge)
    waymo_bin_file = creator.create_whole()
    waymo_subset_bin_file = creator.create_subset()
    print(waymo_bin_file, waymo_subset_bin_file)
    # breakpoint()


if __name__ == '__main__':
    main()
"""
Usage:
python tools/create_gt_bin.py \
    --ann_file ./data/waymo_dev1x/kitti_format/waymo_infos_val.pkl \
    --data_root ./data/waymo_dev1x/waymo_format \
    --split validation \
    --load_interval 1 \
    --for_cam_only_challenge True
"""
