# Modified from https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py # noqa
import imageio
import numpy as np
import os
import struct
import zlib
from argparse import ArgumentParser

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}

COMPRESSION_TYPE_DEPTH = {
    -1: 'unknown',
    0: 'raw_ushort',
    1: 'zlib_ushort',
    2: 'occi_ushort'
}


class RGBDFrame:

    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack('f' * 16, file_handle.read(16 * 4)),
            dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(
            struct.unpack('c' * self.color_size_bytes,
                          file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(
            struct.unpack('c' * self.depth_size_bytes,
                          file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        assert compression_type == 'zlib_ushort'
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        assert compression_type == 'jpeg'
        return imageio.imread(self.color_data)


class SensorData:

    def __init__(self, filename, limit):
        self.version = 4
        self.load(filename, limit)

    def load(self, filename, limit):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = b''.join(
                struct.unpack('c' * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack(
                'i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack(
                'i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            self.frames = []
            if limit > 0 and limit < num_frames:
                index = np.random.choice(
                    np.arange(num_frames), limit, replace=False).tolist()
            else:
                index = list(range(num_frames))
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                if i in index:
                    self.frames.append(frame)

    def export_depth_images(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            depth_data = self.frames[f].decompress_depth(
                self.depth_compression_type)
            depth = np.fromstring(
                depth_data, dtype=np.uint16).reshape(self.depth_height,
                                                     self.depth_width)
            imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)

    def export_color_images(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            color = self.frames[f].decompress_color(
                self.color_compression_type)
            imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%f')

    def export_poses(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            self.save_mat_to_file(self.frames[f].camera_to_world,
                                  os.path.join(output_path,
                                               str(f) + '.txt'))

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('exporting camera intrinsics to', output_path)
        self.save_mat_to_file(self.intrinsic_color,
                              os.path.join(output_path, 'intrinsic_color.txt'))
        self.save_mat_to_file(self.extrinsic_color,
                              os.path.join(output_path, 'extrinsic_color.txt'))
        self.save_mat_to_file(self.intrinsic_depth,
                              os.path.join(output_path, 'intrinsic_depth.txt'))
        self.save_mat_to_file(self.extrinsic_depth,
                              os.path.join(output_path, 'extrinsic_depth.txt'))


def process_scene(path, idx, limit):
    data = SensorData(os.path.join(path, idx, f'{idx}.sens'), limit)
    output_path = os.path.join('posed_images', idx)
    data.export_color_images(output_path)
    data.export_intrinsics(output_path)
    data.export_poses(output_path)


def process_directory(path, limit):
    scenes = os.listdir(path)
    for idx in scenes:
        print(f'extracting {path} {idx}')
        process_scene(path, idx, limit)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max-images-per-scene', type=int, default=300)
    args = parser.parse_args()
    if os.path.exists('scans'):
        process_directory('scans', args.max_images_per_scene)
    if os.path.exists('scans_test'):
        process_directory('scans_test', args.max_images_per_scene)
