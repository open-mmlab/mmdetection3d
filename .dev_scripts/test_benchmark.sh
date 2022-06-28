PARTITION=$1
CHECKPOINT_DIR=$2

echo 'configs/3dssd/3dssd_4x4_kitti-3d-car.py' &
mkdir -p $CHECKPOINT_DIR/configs/3dssd/3dssd_4x4_kitti-3d-car.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION 3dssd_4x4_kitti-3d-car configs/3dssd/3dssd_4x4_kitti-3d-car.py \
$CHECKPOINT_DIR/configs/3dssd/3dssd_4x4_kitti-3d-car.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/3dssd/3dssd_4x4_kitti-3d-car.py/FULL_LOG.txt &

echo 'configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py' &
mkdir -p $CHECKPOINT_DIR/configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py \
$CHECKPOINT_DIR/configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py/FULL_LOG.txt &

echo 'configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py \
$CHECKPOINT_DIR/configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py/FULL_LOG.txt &

echo 'configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py \
$CHECKPOINT_DIR/configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py/FULL_LOG.txt &

echo 'configs/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class configs/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py \
$CHECKPOINT_DIR/configs/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/second/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py/FULL_LOG.txt &

echo 'configs/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d configs/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d.py \
$CHECKPOINT_DIR/configs/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d.py/FULL_LOG.txt &

echo 'configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256.py' &
mkdir -p $CHECKPOINT_DIR/configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION groupfree3d_8x4_scannet-3d-18class-L6-O256 configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256.py \
$CHECKPOINT_DIR/configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256.py/FULL_LOG.txt &

echo 'configs/h3dnet/h3dnet_3x8_scannet-3d-18class.py' &
mkdir -p $CHECKPOINT_DIR/configs/h3dnet/h3dnet_3x8_scannet-3d-18class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION h3dnet_3x8_scannet-3d-18class configs/h3dnet/h3dnet_3x8_scannet-3d-18class.py \
$CHECKPOINT_DIR/configs/h3dnet/h3dnet_3x8_scannet-3d-18class.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/h3dnet/h3dnet_3x8_scannet-3d-18class.py/FULL_LOG.txt &

echo 'configs/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py' &
mkdir -p $CHECKPOINT_DIR/configs/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class configs/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py \
$CHECKPOINT_DIR/configs/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py/FULL_LOG.txt &

echo 'configs/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class.py' &
mkdir -p $CHECKPOINT_DIR/configs/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION imvotenet_stage2_16x8_sunrgbd-3d-10class configs/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class.py \
$CHECKPOINT_DIR/configs/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class.py/FULL_LOG.txt &

echo 'configs/imvoxelnet/imvoxelnet_4x8_kitti-3d-car.py' &
mkdir -p $CHECKPOINT_DIR/configs/imvoxelnet/imvoxelnet_4x8_kitti-3d-car.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION imvoxelnet_4x8_kitti-3d-car configs/imvoxelnet/imvoxelnet_4x8_kitti-3d-car.py \
$CHECKPOINT_DIR/configs/imvoxelnet/imvoxelnet_4x8_kitti-3d-car.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/imvoxelnet/imvoxelnet_4x8_kitti-3d-car.py/FULL_LOG.txt &

echo 'configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py \
$CHECKPOINT_DIR/configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py/FULL_LOG.txt &

echo 'configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py \
$CHECKPOINT_DIR/configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py/FULL_LOG.txt &

echo 'configs/pointnet2/pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class.py' &
mkdir -p $CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class configs/pointnet2/pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class.py \
$CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class.py/latest.pth --eval mIoU \
2>&1|tee $CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class.py/FULL_LOG.txt &

echo 'configs/pointnet2/pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class.py' &
mkdir -p $CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class configs/pointnet2/pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class.py \
$CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/pointnet2/pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class.py/FULL_LOG.txt &

echo 'configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py \
$CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py/latest.pth --format-only --eval-options jsonfile_prefix=$CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py/results_challenge csv_savepath=$CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py/results_challenge.csv \
2>&1|tee $CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py/FULL_LOG.txt &

echo 'configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py \
$CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py/latest.pth --eval waymo --eval-options pklfile_prefix=$CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py/kitti_results submission_prefix=$CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py/kitti_results \
2>&1|tee $CHECKPOINT_DIR/configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py/FULL_LOG.txt &

echo 'configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py \
$CHECKPOINT_DIR/configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py/FULL_LOG.txt &

echo 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py' &
mkdir -p $CHECKPOINT_DIR/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION hv_second_secfpn_6x8_80e_kitti-3d-3class configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py \
$CHECKPOINT_DIR/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py/FULL_LOG.txt &

echo 'configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py' &
mkdir -p $CHECKPOINT_DIR/configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py \
$CHECKPOINT_DIR/configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py/latest.pth --format-only --eval-options jsonfile_prefix=$CHECKPOINT_DIR/configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py/results_challenge csv_savepath=$CHECKPOINT_DIR/configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py/results_challenge.csv \
2>&1|tee $CHECKPOINT_DIR/configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py/FULL_LOG.txt &

echo 'configs/votenet/votenet_8x8_scannet-3d-18class.py' &
mkdir -p $CHECKPOINT_DIR/configs/votenet/votenet_8x8_scannet-3d-18class.py
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=5 ./tools/slurm_test.sh $PARTITION votenet_8x8_scannet-3d-18class configs/votenet/votenet_8x8_scannet-3d-18class.py \
$CHECKPOINT_DIR/configs/votenet/votenet_8x8_scannet-3d-18class.py/latest.pth --eval map \
2>&1|tee $CHECKPOINT_DIR/configs/votenet/votenet_8x8_scannet-3d-18class.py/FULL_LOG.txt &
