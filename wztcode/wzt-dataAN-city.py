import json
import pickle
import os

def load_json_data(filepath):
    """加载JSON文件数据"""
    with open(filepath, 'r') as file:
        return json.load(file)

def load_pkl_data(filepath):
    """加载Pickle文件数据"""
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def determine_city(location):
    """根据location确定城市"""
    if 'boston' in location.lower():
        return 'Boston'
    elif 'singapore' in location.lower():
        return 'Singapore'
    else:
        return 'Unknown'

def find_location_by_token(sample_token, sample_data, scene_data, log_data):
    """根据sample token查找对应的location"""
    scene_token = next((item['scene_token'] for item in sample_data if item['token'] == sample_token), None)
    if scene_token:
        log_token = next((item['log_token'] for item in scene_data if item['token'] == scene_token), None)
        if log_token:
            location = next((item['location'] for item in log_data if item['token'] == log_token), None)
            return location
    return "Location not found"

def main():
    # 确保结果保存文件夹存在
    if not os.path.exists('nuscenes-AnSaveResults'):
        os.makedirs('nuscenes-AnSaveResults')

    # 加载数据
    sample_data = load_json_data('./data/nuscenes/v1.0-trainval/sample.json')
    scene_data = load_json_data('./data/nuscenes/v1.0-trainval/scene.json')
    log_data = load_json_data('./data/nuscenes/v1.0-trainval/log.json')
    tokens_data = load_pkl_data('./data/nuscenes/nuscenes_infos_val.pkl')
    # tokens_data = load_pkl_data('./data/nuscenes/nuscenes_infos_train.pkl')
    
    
    results = []
    city_counts = {'Boston': 0, 'Singapore': 0}
    
    for item in tokens_data['data_list']:
        sample_token = item['token']
        location = find_location_by_token(sample_token, sample_data, scene_data, log_data)
        city = determine_city(location)
        results.append({"token": sample_token, "city": city})
        city_counts[city] += 1
    
    # 将结果保存到文件
    with open('nuscenes-AnSaveResults/val-city.json', 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to 'nuscenes-AnSaveResults/val-city.json'.")
    print(f"Total samples in Boston: {city_counts['Boston']}")
    print(f"Total samples in Singapore: {city_counts['Singapore']}")

if __name__ == "__main__":
    main()
