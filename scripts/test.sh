#!/usr/bin/env bash
set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CONFIG="configs/subset/hv_pointpillars_secfpn_6x2_80e_kitti-3d-car-rgb_paint.py"
WORKDIR="../../checkpoints/3d/hv_pointpillars_secfpn_6x2_80e_kitti-3d-car-rgb_paint"
CHECKPOINT="${WORKDIR}/epoch_40.pth"
RESULT_FILE="${WORKDIR}/epoch_40.pkl"

GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --out ${RESULT_FILE} --eval mAP
