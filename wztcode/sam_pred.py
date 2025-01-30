from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = './data/nuscenes/samples/CAM_BACK/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg'
ckpt = '../PointSAM-for-MixSup/ckpt/sam_vit_h_4b8939.pth'
model_type = 'vit_h'

sam = sam_model_registry[model_type](checkpoint=ckpt)
mask_generator = SamAutomaticMaskGenerator(sam)

image = cv2.imread(img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image)

# 创建一个空白图像用于显示掩码叠加
mask_overlay = image.copy()

# 遍历所有掩码，叠加到图像上
for mask in masks:
    # 获取掩码的二值图
    binary_mask = mask['segmentation']
    
    # 随机生成一个颜色用于当前掩码的显示
    color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
    
    # 将掩码叠加到原始图像（半透明效果）
    mask_overlay[binary_mask] = (
        0.5 * np.array(color) + 0.5 * mask_overlay[binary_mask]
    ).astype(np.uint8)

# 显示叠加结果
plt.figure(figsize=(10, 10))
plt.imshow(mask_overlay)
plt.axis('off')
plt.title("Mask Overlay")
plt.show()

# 保存叠加后的结果到本地
output_path = './mask_overlay_result.jpg'
cv2.imwrite(output_path, cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
print(f"Mask overlay saved to {output_path}")