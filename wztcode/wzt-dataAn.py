from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d
import matplotlib.pyplot as plt
import os

# 初始化nuScenes对象
nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes/', verbose=True)

# 确保输出目录存在
output_dir = 'AnSaveResults'
os.makedirs(output_dir, exist_ok=True)

# 获取特定场景
scene = nusc.get('scene', '3673601f263947fd9259640bd55622bd')
first_sample_token = scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)

# 处理点云数据
lidar_top_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
pcd_path = os.path.join(nusc.dataroot, lidar_top_data['filename'])
pcd = LidarPointCloud.from_file(pcd_path)
points = pcd.points[:3, :]

# 使用Open3D保存点云
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points.T)
o3d.io.write_point_cloud(os.path.join(output_dir, 'Boston-sample_pcd.ply'), cloud)

# 处理所有六个视角的图像数据
camera_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# 遍历所有视角的摄像头
for cam in camera_channels:
    cam_data = nusc.get('sample_data', sample['data'][cam])
    img_path = os.path.join(nusc.dataroot, cam_data['filename'])
    img = plt.imread(img_path)

    # 使用Matplotlib保存每个视角的图像，文件名根据视角命名
    plt.imshow(img)
    plt.axis('off')
    output_image_path = os.path.join(output_dir, f'Boston-{cam}.png')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()

print("点云和六个视角的图像已成功保存到", output_dir)
