import os
import cv2
import time
from tqdm import tqdm
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# 加载预训练模型到 GPU
ckpt = '../PointSAM-for-MixSup/ckpt/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 检查是否有 GPU
print(f"Using device: {device}")

# 加载模型并移动到 GPU
sam = sam_model_registry[model_type](checkpoint=ckpt).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# 定义六个相机视角文件夹名称
camera_views = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

# 数据根目录
data_dir = './data/nuscenes/samples'

# 初始化计时
start_time = time.time()

# 总处理任务数（1000 样本 × 6 个视角）
total_tasks = 1000 * len(camera_views)

# 使用 tqdm 显示进度条
with tqdm(total=total_tasks, desc="Processing Images", unit="task") as pbar:
    # 遍历 1000 个样本
    sample_count = 1000
    for i in range(sample_count):
        for view in camera_views:
            # 构建视角图像路径
            image_dir = os.path.join(data_dir, view)
            image_list = sorted(os.listdir(image_dir))  # 获取视角下的所有图像文件
            
            if i < len(image_list):  # 确保样本索引不超过图像数量
                image_path = os.path.join(image_dir, image_list[i])
                image = cv2.imread(image_path)  # 读取图像
                # image = cv2.resize(image, (64, 64))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB
                
                # 将图像移动到 GPU（如果需要，SAM 的输入通常不需要 tensor 转换）
                masks = mask_generator.generate(image)
            
            # 更新进度条
            pbar.update(1)

# 计算总时间
end_time = time.time()
total_time = end_time - start_time

print(f"处理 {sample_count} 个样本中六个视角图像共耗时：{total_time:.2f} 秒")
print(f"每个样本的平均处理时间：{total_time / sample_count:.2f} 秒")
