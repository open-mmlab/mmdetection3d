import json
import pickle
import os

def find_scene_token(json_data, target_token):
    # 遍历数据，寻找匹配的token
    for record in json_data:
        if record['token'] == target_token:
            return record['scene_token']
    return None

def find_scene_description(scene_data, scene_token):
    # 遍历场景数据，找到对应的场景描述
    for scene in scene_data:
        if scene['token'] == scene_token:
            return scene['description']
    return "No description found."

def main():
    # 读取Pickle文件中的数据
    with open('./data/nuscenes/nuscenes_infos_val.pkl', 'rb') as file:
    # with open('./data/nuscenes/nuscenes_infos_train.pkl', 'rb') as file: 
        nuscenes_data = pickle.load(file)
    
    # 读取sample.json文件
    with open('./data/nuscenes/v1.0-trainval/sample.json', 'r') as file:
        json_data = json.load(file)
    
    # 读取scene.json文件
    with open('./data/nuscenes/v1.0-trainval/scene.json', 'r') as file:
        scene_data = json.load(file)
        
    
    # 初始化计数器
    count_night = 0
    count_rain = 0
    
    # 获取所有target_token并查找对应的scene_token和场景描述
    results = []
    for item in nuscenes_data['data_list']:
        target_token = item['token']
        scene_token = find_scene_token(json_data, target_token)
        if scene_token:
            description = find_scene_description(scene_data, scene_token)
            # 不区分大小写地检查描述中是否包含"night"和"rain"
            if "night" in description.lower() or "rain" in description.lower():
                results.append({
                    "target_token": target_token,
                    "scene_token": scene_token,
                    "description": description
                })
                if "night" in description.lower():
                    count_night += 1
                if "rain" in description.lower():
                    count_rain += 1
    
    # 确保结果保存文件夹存在
    if not os.path.exists('nuscenes-AnSaveResults'):
        os.makedirs('nuscenes-AnSaveResults')
    
    # 将结果保存到文件
    with open('nuscenes-AnSaveResults/val-RainAndDark.json', 'w') as file:
        json.dump(results, file, indent=4)

    # 输出统计信息
    print(f"Total scenes with 'Night': {count_night}")
    print(f"Total scenes with 'Rain': {count_rain}")

if __name__ == "__main__":
    main()
