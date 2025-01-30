import torch
import copy
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS

from projects.BEVFusion.bevfusion.sam_supervised_utils import *

@MODELS.register_module()
class MeanTeacher(Base3DDetector):
    def __init__(self,
                 student_cfg,
                 ema_decay=0.999,
                 data_preprocessor=None,
                 init_cfg=None,
                 **kwargs):
        # Initialize the Base3DDetector with required parameters
        self.voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Initialize the student and teacher models using the BEVFUSION config
        self.student = MODELS.build(student_cfg)
        self.teacher = copy.deepcopy(self.student)

        # Ensure all parameters are float types
        for param in self.student.parameters():
            param.data = param.data.float()
        for param in self.teacher.parameters():
            param.data = param.data.float()

        # Set teacher model's parameters to not require gradients
        for param in self.teacher.parameters():
            param.requires_grad = False

        # EMA decay factor
        self.ema_decay = ema_decay

        self.conv1x1 = torch.nn.Conv2d(in_channels=256, out_channels=80, kernel_size=1, stride=1)

    def extract_feat(self, inputs):
        """Extract features from the inputs using the student model."""
        return self.student.extract_feat(inputs)

    def loss(self, inputs, data_samples, **kwargs):
        """Compute the loss using the student model."""

        return None

    def predict(self, inputs, data_samples, **kwargs):
        """Predict using the student model."""
        return self.student.forward(inputs, data_samples, mode='predict', **kwargs)

    def _forward(self, inputs, data_samples=None, **kwargs):
        """Forward method, which is kept empty as per your request."""
        pass

    def update_teacher(self):
        """Update the teacher model's parameters using EMA of student model."""
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * student_param.data)

    def train_step(self, data, optim_wrapper):
        """Perform a training step for semi-supervised learning."""
        # Separate labeled and unlabeled data

        # labeled_data and unlabeled_data
        labeled_data = data if True in data.get('is_labeled', False) else None
        unlabeled_data = data if True not in data.get('is_labeled', False) else None

        # Process labeled data
        if labeled_data:
            with optim_wrapper.optim_context(self):
                labeled_data = self.data_preprocessor(labeled_data, True)
                labeled_losses, _, _ = self.student.forward(
                    labeled_data['inputs'], labeled_data['data_samples'], mode='loss')

        # Process unlabeled data
        if unlabeled_data:
            with torch.no_grad():
                unlabeled_data = self.data_preprocessor(unlabeled_data, True)
                teacher_outputs = self.teacher.forward(
                    unlabeled_data['inputs'], unlabeled_data['data_samples'], mode='predict'
                )

            # 让 student 推理 unlabeled data
            labeled_losses, student_outputs, before_pts_feature = self.student.forward(
                unlabeled_data['inputs'], unlabeled_data['data_samples'], mode='loss'
            )

            # batch_before_pts_feature = split_sparse_tensor_by_batch(before_pts_feature, before_pts_feature.batch_size)


            # for i, temp_pts_feature in enumerate(batch_before_pts_feature):
            #     batch_before_pts_feature[i]['voxel_centers']=get_voxel_centers(temp_pts_feature['indices'], self.voxelize_cfg["voxel_size"], self.voxelize_cfg["point_cloud_range"][:3])
            #     # batch_before_pts_feature[i]['voxel_centers']=get_voxel_centers(temp_pts_feature['indices'], self.voxelize_cfg["voxel_size"], [0,0,0])

            # for i, temp_pts_feature in enumerate(batch_before_pts_feature):
            #     # image_shape_list = [unlabeled_data['data_samples'][i].batch_input_shape] * 6
            #     image_shape_list = [(1600, 900)] * 6
            #     batch_before_pts_feature[i]['voxel_to_images'] = project_voxel_to_images(temp_pts_feature['voxel_centers'], unlabeled_data['data_samples'][i].lidar2img, image_shape_list, unlabeled_data['data_samples'][i].lidar_aug_matrix)

            # save_voxel_centers(unlabeled_data['inputs']['points'][0][:,:3],"/data/otherg/mmdetection3d/wztcode/points.ply")

            # 调整 before_pts_feature[1] 的通道数以匹配 before_pts_feature[0]

            before_pts_feature_resized_1 = self.conv1x1(before_pts_feature[1])

            # 计算 before_pts_feature 的对齐损失 (余弦相似度损失)
            cos_sim_1 = torch.nn.functional.cosine_similarity(
                before_pts_feature[0].flatten(start_dim=1),
                before_pts_feature[2].flatten(start_dim=1),
                dim=1
            )
            loss_1 = 1 - cos_sim_1.mean()
            
            cos_sim_2 = torch.nn.functional.cosine_similarity(
                before_pts_feature[0].flatten(start_dim=1),
                before_pts_feature_resized_1.flatten(start_dim=1),
                dim=1
            )
            loss_2 = 1 - cos_sim_2.mean()

            # 累加损失
            labeled_losses['alignment_loss'] = loss_1 + loss_2

            consistency_loss = 0
            for i, (student_sample, teacher_sample) in enumerate(zip(student_outputs, teacher_outputs)):
                if 'pred_instances_3d' in teacher_sample:
                    student_boxes = student_sample.pred_instances_3d.bboxes_3d.tensor
                    teacher_boxes = teacher_sample.pred_instances_3d.bboxes_3d.tensor
                    # 直接赋值一致性损失
                    consistency_loss += torch.nn.functional.mse_loss(student_boxes, teacher_boxes) * 0.1

            labeled_losses['consistency_loss'] = labeled_losses.get('consistency_loss', 0) + consistency_loss
        # 聚合损失并更新模型
        parsed_losses, log_vars = self.parse_losses(labeled_losses)
        optim_wrapper.update_params(parsed_losses)

        # Update teacher model parameters using EMA
        self.update_teacher()

        return log_vars
