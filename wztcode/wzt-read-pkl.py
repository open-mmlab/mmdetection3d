import pickle

# 定义文件路径
# file_path = './data/nuscenes/nuscenes_dbinfos_train.pkl'
# file_path = './data/nuscenes/nuscenes_infos_train_singapore_notrain_nonight.pkl'
file_path = './data/nuscenes/nuscenes_infos_val_singapore_boston_rain_night_traget.pkl'
# file_path = './data/nuscenes/nuscenes_infos_val_sampled0125.pkl'


# 读取 .pkl 文件
with open(file_path, 'rb') as f:
    nuscenes_infos = pickle.load(f)

# 打印读取的数据结构
# print(type(nuscenes_infos))  # 查看读取的数据类型
# print(len(nuscenes_infos))   # 查看数据量

print(nuscenes_infos['metainfo'])
print(nuscenes_infos['data_list'][0])
# 查看文件内容的前几个元素
# for i, info in enumerate(nuscenes_infos[:5]):  # 只查看前5个元素
#     print(f"Info {i+1}: {info}")
