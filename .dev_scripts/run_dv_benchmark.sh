PARTITION=$1
CHECKPOINT_DIR=$2

echo './configs/dynamic_voxelization/dv_second_secfpn_6x8_80e_kitti-3d-car.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION dv_second_secfpn_6x8_80e_kitti-3d-car ./configs/dynamic_voxelization/dv_second_secfpn_6x8_80e_kitti-3d-car.py \
$CHECKPOINT_DIR/dv_second_secfpn_6x8_80e_kitti-3d-car --cfg-options checkpoint_config.max_keep_ckpts=7 >/dev/null &

echo './configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class ./configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py \
$CHECKPOINT_DIR/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class --cfg-options checkpoint_config.max_keep_ckpts=7 >/dev/null &

echo './configs/dynamic_voxelization/dv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION dv_pointpillars_secfpn_6x8_160e_kitti-3d-car ./configs/dynamic_voxelization/dv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py \
$CHECKPOINT_DIR/dv_pointpillars_secfpn_6x8_160e_kitti-3d-car --cfg-options checkpoint_config.max_keep_ckpts=7 >/dev/null &
