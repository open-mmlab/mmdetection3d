#!/usr/bin/env bash

DOWNLOAD_DIR=$1  # 数据集下载路径，mim 会传入 dataset-index.yml 中的 download_root
DATA_ROOT=$2  # 数据集存放路径，mim 会传入 dataset-index.yml 中的 data_root

#  解包，预处理命令
cat $DOWNLOAD_DIR/kitti/raw/*.tar.gz.*  | tar -xvz -C $DATA_ROOT/..
tar -xvf $DATA_ROOT/kitti.tar -C $DATA_ROOT/..
rm $DATA_ROOT/kitti.tar
