PARTITION=$1
CHECKPOINT_DIR=$2

echo 'configs/3dssd/3dssd_4xb4_kitti-3d-car.py' &
mkdir -p $CHECKPOINT_DIR/configs/3dssd/3dssd_4xb4_kitti-3d-car.py
GPUS=4 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION 3dssd_4x4_kitti-3d-car configs/3dssd/3dssd_4xb4_kitti-3d-car.py \
$CHECKPOINT_DIR/configs/3dssd/3dssd_4xb4_kitti-3d-car.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/3dssd/3dssd_4xb4_kitti-3d-car.py/FULL_LOG.txt &

echo 'configs/centerpoint/centerpoint_pillar02_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus configs/centerpoint/centerpoint_pillar02_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py \
$CHECKPOINT_DIR/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py/FULL_LOG.txt &

echo 'configs/dynamic_voxelization/second_dv_secfpn_8xb2-cosine-80e_kitti-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/dynamic_voxelization/second_dv_secfpn_8xb2-cosine-80e_kitti-3d-3class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class configs/dynamic_voxelization/second_dv_secfpn_8xb2-cosine-80e_kitti-3d-3class.py \
$CHECKPOINT_DIR/configs/dynamic_voxelization/second_dv_secfpn_8xb2-cosine-80e_kitti-3d-3class.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/dynamic_voxelization/second_dv_secfpn_8xb2-cosine-80e_kitti-3d-3class.py/FULL_LOG.txt &

echo 'configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/fcos3d/fcos3d_r101-caffe-fpn-head-gn-dcn_8xb2-1x_nus-mono3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d configs/fcos3d/fcos3d_r101-caffe-fpn-head-gn-dcn_8xb2-1x_nus-mono3d.py \
$CHECKPOINT_DIR/configs/fcos3d/fcos3d_r101-caffe-fpn-head-gn-dcn_8xb2-1x_nus-mono3d.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/fcos3d/fcos3d_r101-caffe-fpn-head-gn-dcn_8xb2-1x_nus-mono3d.py/FULL_LOG.txt &

echo 'configs/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class configs/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py \
$CHECKPOINT_DIR/configs/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py/FULL_LOG.txt &

echo 'configs/free_anchor/pointpillars_hv_regnet-1.6gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/free_anchor/pointpillars_hv_regnet-1.6gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d configs/free_anchor/pointpillars_hv_regnet-1.6gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d.py \
$CHECKPOINT_DIR/configs/free_anchor/pointpillars_hv_regnet-1.6gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/free_anchor/pointpillars_hv_regnet-1.6gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d.py/FULL_LOG.txt &

echo 'configs/groupfree3d/groupfree3d_head-L6-O256_4xb8_scannet-seg.py' &
mkdir -p $CHECKPOINT_DIR/configs/groupfree3d/groupfree3d_head-L6-O256_4xb8_scannet-seg.py
GPUS=4 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION groupfree3d_8x4_scannet-3d-18class-L6-O256 configs/groupfree3d/groupfree3d_head-L6-O256_4xb8_scannet-seg.py \
$CHECKPOINT_DIR/configs/groupfree3d/groupfree3d_head-L6-O256_4xb8_scannet-seg.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/groupfree3d/groupfree3d_head-L6-O256_4xb8_scannet-seg.py/FULL_LOG.txt &

echo 'configs/h3dnet/h3dnet_8xb3_scannet-seg.py' &
mkdir -p $CHECKPOINT_DIR/configs/h3dnet/h3dnet_8xb3_scannet-seg.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION h3dnet_3x8_scannet-3d-18class configs/h3dnet/h3dnet_8xb3_scannet-seg.py \
$CHECKPOINT_DIR/configs/h3dnet/h3dnet_8xb3_scannet-seg.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/h3dnet/h3dnet_8xb3_scannet-seg.py/FULL_LOG.txt &

echo 'configs/imvotenet/imvotenet_faster-rcnn-r50_fpn_4xb2_sunrgbd-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/imvotenet/imvotenet_faster-rcnn-r50_fpn_4xb2_sunrgbd-3d.py
GPUS=4 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class configs/imvotenet/imvotenet_faster-rcnn-r50_fpn_4xb2_sunrgbd-3d.py \
$CHECKPOINT_DIR/configs/imvotenet/imvotenet_faster-rcnn-r50_fpn_4xb2_sunrgbd-3d.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/imvotenet/imvotenet_faster-rcnn-r50_fpn_4xb2_sunrgbd-3d.py/FULL_LOG.txt &

echo 'configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION imvotenet_stage2_16x8_sunrgbd-3d-10class configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py \
$CHECKPOINT_DIR/configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py/FULL_LOG.txt &

echo 'configs/imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py' &
mkdir -p $CHECKPOINT_DIR/configs/imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION imvoxelnet_4x8_kitti-3d-car configs/imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py \
$CHECKPOINT_DIR/configs/imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py/FULL_LOG.txt &

echo 'configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py \
$CHECKPOINT_DIR/configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py/FULL_LOG.txt &

echo 'configs/parta2/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/parta2/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class configs/parta2/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class.py \
$CHECKPOINT_DIR/configs/parta2/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/parta2/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class.py/FULL_LOG.txt &

echo 'configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_s3dis-seg.py' &
mkdir -p $CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_s3dis-seg.py
GPUS=2 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_s3dis-seg.py \
$CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_s3dis-seg.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_s3dis-seg.py/FULL_LOG.txt &

echo 'configs/pointnet2/pointnet2_msg_2xb16-cosine-250e_scannet-seg.py' &
mkdir -p $CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_2xb16-cosine-250e_scannet-seg.py
GPUS=2 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class configs/pointnet2/pointnet2_msg_2xb16-cosine-250e_scannet-seg.py \
$CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_2xb16-cosine-250e_scannet-seg.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_2xb16-cosine-250e_scannet-seg.py/FULL_LOG.txt &

echo 'configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d.py \
$CHECKPOINT_DIR/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d.py/FULL_LOG.txt &

echo 'configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py
GPUS=16 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py \
$CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py/FULL_LOG.txt &

echo 'configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py \
$CHECKPOINT_DIR/configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py/FULL_LOG.txt &

echo 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION hv_second_secfpn_6x8_80e_kitti-3d-3class configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py \
$CHECKPOINT_DIR/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py/FULL_LOG.txt &

echo 'configs/ssn/ssn_hv_secfpn_sbn-all_16xb2-2x_lyft-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/ssn/ssn_hv_secfpn_sbn-all_16xb2-2x_lyft-3d.py
GPUS=16 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d configs/ssn/ssn_hv_secfpn_sbn-all_16xb2-2x_lyft-3d.py \
$CHECKPOINT_DIR/configs/ssn/ssn_hv_secfpn_sbn-all_16xb2-2x_lyft-3d.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/ssn/ssn_hv_secfpn_sbn-all_16xb2-2x_lyft-3d.py/FULL_LOG.txt &

echo 'configs/votenet/votenet_8xb8_scannet-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/votenet/votenet_8xb8_scannet-3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_train.sh $PARTITION votenet_8x8_scannet-3d-18class configs/votenet/votenet_8xb8_scannet-3d.py \
$CHECKPOINT_DIR/configs/votenet/votenet_8xb8_scannet-3d.py --cfg-options checkpoint_config.max_keep_ckpts=1 \
2>&1|tee $CHECKPOINT_DIR/configs/votenet/votenet_8xb8_scannet-3d.py/FULL_LOG.txt &
