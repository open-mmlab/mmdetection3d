import pickle
import numpy

car_cnt=0
cyc_cnt=0
ped_cnt=0

with open('../../data/rf2021/rf2021_infos_train.pkl','rb') as f:
	datas=pickle.load(f)

for data in datas:
	for label in data['annos']['gt_names']:
		if label=='Car':
			car_cnt+=1
		elif label=='Cyclist':
			cyc_cnt+=1
		elif label=='Pedestrian':
			ped_cnt+=1

print(car_cnt, cyc_cnt, ped_cnt)
