import pickle
import json


torv_split='val'

# 定义文件路径
nuscenes_infos_file = f'./data/nuscenes/nuscenes_infos_{torv_split}.pkl'
train_city_file = f'./wztcode/{torv_split}-city.json'
train_rain_night_file = f'./wztcode/{torv_split}-RainAndDark.json'
output_file = f'./data/nuscenes/nuscenes_infos_{torv_split}_singapore_boston_rain_night_traget.pkl'



# 读取 .pkl 文件
with open(nuscenes_infos_file, 'rb') as f:
    nuscenes_infos = pickle.load(f)

# 读取 train-city.json 文件
with open(train_city_file, 'r') as f:
    train_city_data = json.load(f)

# 读取 train-RainAndDark.json 文件
with open(train_rain_night_file, 'r') as f:
    train_rain_night_data = json.load(f)

# 获取 Singapore 的 sample tokens
singapore_tokens = [sample['token'] for sample in train_city_data if sample['city'] == 'Singapore']

# 获取下雨或夜晚的 sample tokens
rain_night_tokens = [sample['target_token'] for sample in train_rain_night_data]

# 筛选出除了 Singapore、非下雨、非夜晚 的所有样本
filtered_data_list = [
    sample for sample in nuscenes_infos['data_list']
    if sample['token'] not in singapore_tokens or sample['token'] in rain_night_tokens
]

# 创建新的 nuscenes_infos 结构并保存
filtered_nuscenes_infos = {
    'metainfo': nuscenes_infos['metainfo'],
    'data_list': filtered_data_list
}

# 保存到新的 .pkl 文件
with open(output_file, 'wb') as f:
    pickle.dump(filtered_nuscenes_infos, f)

print(f"筛选完成，共保存 {len(filtered_data_list)} 个样本到 {output_file}")
