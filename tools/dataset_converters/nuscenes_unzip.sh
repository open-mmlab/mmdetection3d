#!/usr/bin/env bash

DOWNLOAD_DIR=$1  # The directory where the downloaded data set is stored
DATA_ROOT=$2  # The root directory of the converted dataset

for split in $DOWNLOAD_DIR/nuScenes/raw/*; do
    for tgz_file in $split/*; do
        if [[ $tgz_file == *.tgz ]]
        then
            echo "Unzipping $tgz_file to $DATA_ROOT ......"
            unzip -oq $tgz_file -d $DATA_ROOT/
            echo "[Done] Unzip $tgz_file to $DATA_ROOT"
        fi
        # delete the original files
        rm -f $tgz_file
    done
done
