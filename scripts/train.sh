#!/usr/bin/env bash
set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CONFIG="configs/subset/hv_pointpillars_secfpn_6x2_80e_kitti-3d-car-rgb_paint.py"
WORKDIR="/home/thuync/checkpoints/3d/hv_pointpillars_secfpn_6x2_80e_kitti-3d-car-rgb_paint"

GPUS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/train.py $CONFIG --launcher pytorch --work-dir $WORKDIR --autoscale-lr
