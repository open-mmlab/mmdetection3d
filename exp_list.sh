#!/bin/bash
CONFIG_FILES=(
mvxnet_efficiency_es_fpn_second_fpn_kitti-3d-3class
mvxnet_efficiency_es_fpn_squeeze_fpn_kitti-3d-3class
mvxnet_efficiency_fpn_second_fpn_kitti-3d-3class
mvxnet_efficiency_fpn_squeeze_fpn_kitti-3d-3class
mvxnet_fpn_dv_second_secfpn_320x92_kitti-3d-3class
mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class
mvxnet_fpn_dv_second_squeezefpn_320x92_kitti-3d-3class
mvxnet_fpn_dv_second_squeezefpn_8xb2-80e_kitti-3d-3class
mvxnet_mobilenetv2_fpn_second_fpn_kitti-3d-3class
mvxnet_sqeezefpn_secfpn_kitti-3d-3class
mvxnet_sqeezefpn_secfpn_2x_scale_kitti-3d-3class
)

TIMESTAMP=$(TZ='America/Los_Angeles' date +%m%d)
for item in "${CONFIG_FILES[@]}"; do
  echo "############"
  echo
  workdir="work_dirs/"${item}
  mkdir -p ${workdir}
  config="configs/mvxnet/"${item}.py
  saved_model=$workdir/epoch_20.pth
  echo "Config:" $config
  echo "Workdir:" $workdir
  echo "Saved Model:" $saved_model

  echo "Train Command:"
  echo "python tools/train.py $config 2>&1 | tee $workdir/train-${TIMESTAMP}.log"
  echo "Test Command:"
  echo "python tools/test.py $workdir/$item.py $saved_model 2>&1 | tee $workdir/test-${TIMESTAMP}.log"

done
