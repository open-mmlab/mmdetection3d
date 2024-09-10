#!/usr/bin/env bash

DOWNLOAD_DIR=$1  # The directory where the downloaded data set is stored
DATA_ROOT=$2  # The root directory of the converted dataset

for zip_file in $DOWNLOAD_DIR/KITTI_Object/raw/*.zip; do
    echo "Unzipping $zip_file to $DATA_ROOT ......"
	unzip -oq $zip_file -d $DATA_ROOT
    echo "[Done] Unzip $zip_file to $DATA_ROOT"
    # delete the original files
	rm -f $zip_file
done
