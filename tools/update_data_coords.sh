#!/usr/bin/env bash

set -x
export PYTHONPATH=`pwd`:$PYTHONPATH

PARTITION=$1
DATASET=$2
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
JOB_NAME=update_data_coords

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/update_data_coords.py ${DATASET} \
            --root-dir ./data/${DATASET} \
            --out-dir ./data/${DATASET}
