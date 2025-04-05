import os
import random
import shutil
from os.path import join as osp


root_dir = '../data/Car'
points_dir = os.path.join(root_dir, 'source_data/points')
labels_dir = os.path.join(root_dir, 'source_data/labels')
image_sets_dir = os.path.join(root_dir, 'ImageSets')
dst_training_dir = os.path.join(root_dir, 'training')
dst_testing_dir = os.path.join(root_dir, 'testing')


os.makedirs(image_sets_dir, exist_ok=True)


points_files = os.listdir(points_dir)
labels_files = os.listdir(labels_dir)


random.shuffle(points_files)


total_files = len(points_files)
train_ratio = 0.8
val_ratio = 0.1


train_split = int(total_files * train_ratio)
val_split = train_split + int(total_files * val_ratio)


with open(os.path.join(image_sets_dir, 'train.txt'), 'w') as train_file:
    for filename in points_files[:train_split]:
        file_prefix = os.path.splitext(filename)[0]
        shutil.copy(osp(points_dir, file_prefix + '.bin'), osp(dst_training_dir, 'velodyne'))
        shutil.copy(osp(labels_dir, file_prefix + '.txt'), osp(dst_training_dir, 'label_2'))
        train_file.write(file_prefix + '\n')

with open(os.path.join(image_sets_dir, 'val.txt'), 'w') as val_file:
    for filename in points_files[train_split:val_split]:
        file_prefix = os.path.splitext(filename)[0]
        shutil.copy(osp(points_dir, file_prefix + '.bin'), osp(dst_training_dir, 'velodyne'))
        shutil.copy(osp(labels_dir, file_prefix + '.txt'), osp(dst_training_dir, 'label_2'))
        val_file.write(file_prefix + '\n')

with open(os.path.join(image_sets_dir, 'test.txt'), 'w') as test_file:
    for filename in points_files[val_split:]:
        file_prefix = os.path.splitext(filename)[0]
        shutil.copy(osp(points_dir, file_prefix + '.bin'), osp(dst_testing_dir, 'velodyne'))
        shutil.copy(osp(labels_dir, file_prefix + '.txt'), osp(dst_testing_dir, 'label_2'))
        test_file.write(file_prefix + '\n')
