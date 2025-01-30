from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 配置模型
model_type = "vit_h"
checkpoint = "../PointSAM-for-MixSup/ckpt/sam_vit_h_4b8939.pth"  # 替换为你的模型权重路径
sam = sam_model_registry[model_type](checkpoint=checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)

# 载入图片
image_path = "./data/nuscenes/samples/CAM_BACK/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 生成掩码
masks = mask_generator.generate(image)

# 初始化语义分割图
h, w, _ = image.shape
semantic_map = np.zeros((h, w), dtype=np.int32)

# 示例：模拟类别分配（根据具体任务替换为检测模型输出）
# 假设类别映射：0=背景，1=car，2=pedestrian，3=tree...
category_mapping = {0: "background", 1: "car", 2: "pedestrian", 3: "tree"}
instance_to_category = {0: 1, 1: 2, 2: 3}  # 手动映射实例到类别的例子

# 遍历掩码并分配类别
for idx, mask in enumerate(masks):
    segmentation = mask["segmentation"]
    category = instance_to_category.get(idx, 0)  # 获取类别（默认0=背景）
    semantic_map[segmentation] = category

# 可视化语义分割图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(semantic_map, cmap="jet")
plt.title("Semantic Segmentation")
plt.axis("off")
plt.show()

# 保存语义分割图到本地
output_path = "./semantic_segmentation_result.png"
cv2.imwrite(output_path, semantic_map)
print(f"Semantic segmentation result saved to {output_path}")
