#!/usr/bin/env bash
set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CONFIG="configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py"
WORKDIR="/home/thuync/checkpoints/3d"
CHECKPOINT="${WORKDIR}/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20200620_230614-77663cd6.pth"
RESULT_FILE="${WORKDIR}/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20200620_230614-77663cd6.pkl"

GPUS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --out ${RESULT_FILE} --eval mAP
