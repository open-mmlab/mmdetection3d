import torch
from PIL import Image
from torchvision import transforms
from segment_anything import sam_model_registry

# 设置路径和模型信息
img = '../data/nuscenes/samples/CAM_BACK/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg'
ckpt = './SAM_code/sam_vit_h_4b8939.pth'
model_type = 'vit_h'  # 选择的模型类型

# 1. 加载预训练的 SAM 模型
print("Loading SAM model...")
sam = sam_model_registry[model_type](checkpoint=ckpt)

# 将模型切换到评估模式
sam.eval()

# 2. 图像预处理
print("Processing image...")
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # 调整图像大小
    transforms.ToTensor(),           # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

# 打开图像并应用预处理
image = Image.open(img).convert("RGB")  # 确保图像是RGB格式
input_image = transform(image).unsqueeze(0)  # 增加批次维度，形状为 (1, C, H, W)

# 3. 提取中间特征
print("Extracting image features...")
with torch.no_grad():
    image_features = sam.image_encoder(input_image)
    print("Image Features Shape:", image_features.shape)  # 输出特征形状

# 4. 保存特征到文件（可选）
output_path = "./image_features.pt"
torch.save(image_features, output_path)
print(f"Image features saved to '{output_path}'")

# 5. 转换特征为图片
# 假设提取的特征形状为 (1, C, H, W)，我们需要选取或整合通道维度 C
feature_map = image_features.squeeze(0).mean(dim=0)  # 对通道维度 C 进行平均，得到 (H, W)

# 将特征归一化到 [0, 255] 范围
feature_map_normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
feature_map_image = (feature_map_normalized * 255).byte().cpu().numpy()

# 保存特征图为图片
feature_image = Image.fromarray(feature_map_image).convert("L")  # 转为灰度图
output_image_path = "./image_features_visualization.png"
feature_image.save(output_image_path)
print(f"Feature visualization saved to '{output_image_path}'")