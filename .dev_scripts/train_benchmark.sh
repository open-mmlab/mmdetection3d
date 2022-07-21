PARTITION=$1
WORK_DIR=$2
CPUS_PER_TASK=${3:-4}

echo 'configs/3dssd/3dssd_4x4_kitti-3d-car.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION 3dssd_4x4_kitti-3d-car configs/3dssd/3dssd_4x4_kitti-3d-car.py $WORK_DIR/3dssd_4x4_kitti-3d-car --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py $WORK_DIR/centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class configs/dgcnn/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class.py $WORK_DIR/dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py $WORK_DIR/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py $WORK_DIR/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d configs/free_anchor/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d.py $WORK_DIR/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION groupfree3d_8x4_scannet-3d-18class-L6-O256 configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256.py $WORK_DIR/groupfree3d_8x4_scannet-3d-18class-L6-O256 --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/h3dnet/h3dnet_3x8_scannet-3d-18class.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION h3dnet_3x8_scannet-3d-18class configs/h3dnet/h3dnet_3x8_scannet-3d-18class.py $WORK_DIR/h3dnet_3x8_scannet-3d-18class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class configs/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py $WORK_DIR/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION imvotenet_stage2_16x8_sunrgbd-3d-10class configs/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class.py $WORK_DIR/imvotenet_stage2_16x8_sunrgbd-3d-10class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/imvoxelnet/imvoxelnet_4x8_kitti-3d-car.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION imvoxelnet_4x8_kitti-3d-car configs/imvoxelnet/imvoxelnet_4x8_kitti-3d-car.py $WORK_DIR/imvoxelnet_4x8_kitti-3d-car --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py $WORK_DIR/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/nuimages/mask_rcnn_r50_caffe_fpn_1x_nuim.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION mask_rcnn_r50_caffe_fpn_1x_nuim configs/nuimages/mask_rcnn_r50_caffe_fpn_1x_nuim.py $WORK_DIR/mask_rcnn_r50_caffe_fpn_1x_nuim --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/paconv/paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class configs/paconv/paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class.py $WORK_DIR/paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py $WORK_DIR/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d configs/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d.py $WORK_DIR/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pgd/pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d configs/pgd/pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d.py $WORK_DIR/pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/point_rcnn/point_rcnn_2x8_kitti-3d-3classes.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION point_rcnn_2x8_kitti-3d-3classes configs/point_rcnn/point_rcnn_2x8_kitti-3d-3classes.py $WORK_DIR/point_rcnn_2x8_kitti-3d-3classes --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pointnet2/pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class.py' &
GPUS=2  GPUS_PER_NODE=2  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class configs/pointnet2/pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class.py $WORK_DIR/pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pointnet2/pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class.py' &
GPUS=2  GPUS_PER_NODE=2  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class configs/pointnet2/pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class.py $WORK_DIR/pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d configs/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py $WORK_DIR/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py $WORK_DIR/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py $WORK_DIR/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/sassd/sassd_6x8_80e_kitti-3d-3class.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION sassd_6x8_80e_kitti-3d-3class configs/sassd/sassd_6x8_80e_kitti-3d-3class.py $WORK_DIR/sassd_6x8_80e_kitti-3d-3class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION hv_second_secfpn_6x8_80e_kitti-3d-3class configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py $WORK_DIR/hv_second_secfpn_6x8_80e_kitti-3d-3class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/second/hv_second_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION hv_second_secfpn_sbn_2x16_2x_waymoD5-3d-3class configs/second/hv_second_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py $WORK_DIR/hv_second_secfpn_sbn_2x16_2x_waymoD5-3d-3class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d configs/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py $WORK_DIR/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py $WORK_DIR/hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d.py' &
GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d configs/ssn/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d.py $WORK_DIR/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/votenet/votenet_8x8_scannet-3d-18class.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION votenet_8x8_scannet-3d-18class configs/votenet/votenet_8x8_scannet-3d-18class.py $WORK_DIR/votenet_8x8_scannet-3d-18class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
echo 'configs/votenet/votenet_16x8_sunrgbd-3d-10class.py' &
GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=$CPUS_PRE_TASK ./tools/slurm_train.sh $PARTITION votenet_16x8_sunrgbd-3d-10class configs/votenet/votenet_16x8_sunrgbd-3d-10class.py $WORK_DIR/votenet_16x8_sunrgbd-3d-10class --cfg-options default_hooks.checkpoint.max_keep_ckpts=1 &
