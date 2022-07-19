import pickle
import numpy as np
import open3d as o3d
from pypcd import pypcd

def extract_ped(datas,ped_cnt,pcd_cnt):
	result=[]
	for data in datas:
		ped_idx=np.where(data['annos']['gt_names']=='Pedestrian')
		if (len(ped_idx[0])>=ped_cnt):
			cur_data=data
			cur_data['annos']['gt_bboxes_3d']=np.delete(cur_data['annos']['gt_bboxes_3d'],range(ped_idx[0][0]),axis=0)
			cur_data['annos']['gt_names']=np.delete(cur_data['annos']['gt_names'],range(ped_idx[0][0]),axis=0)
			result.append(cur_data)
			if len(result)==pcd_cnt:
				break
	return result

def extract_ped_pointless(datas,point_cnt):
    result=[]
    cnt=0
    for data in datas:
        cnt+=1
        if cnt%100==0:
            print(cnt)
        cur_data=data
        delete_list=[]
        for i in range(len(cur_data['annos']['gt_bboxes_3d'])):
            center = cur_data['annos']['gt_bboxes_3d'][i, 0:3]
            dim = cur_data['annos']['gt_bboxes_3d'][i, 3:6]
            yaw = np.zeros(3)
            yaw[2] = cur_data['annos']['gt_bboxes_3d'][i, 6]
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)

            center[2] += dim[2] / 2
            box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)

            pcd = o3d.io.read_point_cloud(cur_data['lidar_points']['lidar_path'])
            
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            if len(indices)<point_cnt:
                delete_list.append(i)
        cur_data['annos']['gt_bboxes_3d']=np.delete(cur_data['annos']['gt_bboxes_3d'],delete_list,axis=0)
        cur_data['annos']['gt_names']=np.delete(cur_data['annos']['gt_names'],delete_list,axis=0)
        result.append(cur_data)
    return result


with open('../../data/rf2021/rf2021_infos_train.pkl','rb') as f:
	datas=pickle.load(f)

ped_cnt=50
point_cnt=30
pcd_cnt=5000000
ped_datas=extract_ped(datas,ped_cnt,pcd_cnt)
ped_datas=extract_ped_pointless(ped_datas,point_cnt)

ped=0
max_ped=0
for data in ped_datas:
    ped+=len(data['annos']['gt_names'])
    max_ped=max_ped if (max_ped>len(data['annos']['gt_names'])) else len(data['annos']['gt_names'])

print(ped_cnt, point_cnt, pcd_cnt, len(ped_datas), ped, max_ped)


with open('../../data/rf2021/rf2021_infos_train_ped_'+str(ped_cnt)+'_points_'+str(point_cnt)+'.pkl','wb') as rf:
	pickle.dump(ped_datas,rf,protocol=pickle.HIGHEST_PROTOCOL)