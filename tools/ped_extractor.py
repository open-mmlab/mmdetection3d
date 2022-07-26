import pickle
import numpy

def extract_ped(datas,ped_cnt,data_cnt):
	result=[]
	for data in datas:
		ped_idx=numpy.where(data['annos']['gt_names']=='Pedestrian')
		if (len(ped_idx[0])>=ped_cnt):
			cur_data=data
			cur_data['annos']['gt_bboxes_3d']=numpy.delete(cur_data['annos']['gt_bboxes_3d'],range(ped_idx[0][0]),axis=0)
			cur_data['annos']['gt_names']=numpy.delete(cur_data['annos']['gt_names'],range(ped_idx[0][0]),axis=0)
			ped_range_idx=[]
			for i in range(len(cur_data['annos']['gt_bboxes_3d'])):
				if cur_data['annos']['gt_bboxes_3d'][i][1] < -80 or cur_data['annos']['gt_bboxes_3d'][i][1] > 50:
					ped_range_idx.append(i)
			cur_data['annos']['gt_bboxes_3d']=numpy.delete(cur_data['annos']['gt_bboxes_3d'],ped_range_idx,axis=0)
			cur_data['annos']['gt_names']=numpy.delete(cur_data['annos']['gt_names'],ped_range_idx,axis=0)
			result.append(cur_data)
			if len(result)==data_cnt:
				break
	return result

if __name__ == '__main__':
	filepath = './data/rf2021/'
	train_data_path = './data/rf2021/' + 'rf2021_infos_train.pkl'

	with open(train_data_path,'rb') as f:
		datas=pickle.load(f)

	ped_cnt= 10
	data_cnt=1000000
	ped_data=extract_ped(datas,ped_cnt,data_cnt)


	with open('./data/rf2021/rf2021_infos_train_only_ped'+str(ped_cnt)+'.pkl','wb') as rf:
		pickle.dump(ped_data,rf,protocol=pickle.HIGHEST_PROTOCOL)
