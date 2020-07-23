#!/usr/bin/env bash
set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

DATA_ROOT='/dataset/kitti'
IMG_SET_DIR="${DATA_ROOT}/ImageSets"

if [ -d "$DIRECTORY" ]; then
	mkdir ${IMG_SET_DIR}
fi

wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O "${IMG_SET_DIR}/test.txt"
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O "${IMG_SET_DIR}/train.txt"
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O "${IMG_SET_DIR}/val.txt"
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O "${IMG_SET_DIR}/trainval.txt"

python tools/create_data.py kitti --root-path ${DATA_ROOT} --out-dir ${DATA_ROOT} --extra-tag kitti #--version mask
