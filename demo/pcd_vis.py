import numpy as np
import open3d as o3d

if __name__ == "__main__":
    dataset = 'NUSCENES'
    if dataset == 'NUSCENES':
        data_feature = 5
    elif dataset == 'KITTI':
        data_feature = 4
    else :
        print('unknow dataset!')

    # pts_filename = "../dataset/nuscenes_sample.bin"
    # pts_filename = "../_out_10hz_8ft_0pitch_withID_labelRange50x50/training/velodyne/012830.bin"
    pts_filename = "../data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151066447194.pcd.bin"
    pcd = np.fromfile(pts_filename, dtype=np.float32).reshape(-1, data_feature)

    # print(pcd)
    pcd_xyz = pcd[:, :3]

    # print(pcd_xyz)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_xyz)
    o3d.visualization.draw_geometries([pcd_o3d])
    #
    # print("Downsample the point cloud with a voxel of 0.05")
    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    # o3d.visualization.draw_geometries([downpcd])
    #
    # print("Recompute the normal of the downsampled point cloud")
    # downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.1, max_nn=30))
    # o3d.visualization.draw_geometries([downpcd])
    #
    # print("Print a normal vector of the 0th point")
    # print(downpcd.normals[0])
    # print("Print the normal vectors of the first 10 points")
    # print(np.asarray(downpcd.normals)[:10, :])
    # print("")
    #
    # print("Load a polygon volume and use it to crop the original point cloud")
    # vol = o3d.visualization.read_selection_polygon_volume(
    #     "../../TestData/Crop/cropped.json")
    # chair = vol.crop_point_cloud(pcd)
    # o3d.visualization.draw_geometries([chair])
    # print("")
    #
    # print("Paint chair")
    # chair.paint_uniform_color([1, 0.706, 0])
    # o3d.visualization.draw_geometries([chair])
    # print("")