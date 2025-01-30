import pickle
import os

def split_nuscenes_data(file_type):
    file_path = f'../../data/nuscenes/nuscenes_mini_infos_{file_type}.pkl'

    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Calculate the split indices based on 1:4 ratio
    total_size = len(data['data_list'])
    source_size = total_size // 5  # 1/5 of the data
    target_size = total_size - source_size  # Remaining 4/5 of the data

    source_data = {
        'metainfo': data['metainfo'],
        'data_list': data['data_list'][:source_size]
    }
    target_data = {
        'metainfo': data['metainfo'],
        'data_list': data['data_list'][source_size:]
    }

    source_file_path = f'../../data/nuscenes/nuscenes_mini_infos_{file_type}_source.pkl'
    target_file_path = f'../../data/nuscenes/nuscenes_mini_infos_{file_type}_target.pkl'

    with open(source_file_path, 'wb') as f:
        pickle.dump(source_data, f)

    with open(target_file_path, 'wb') as f:
        pickle.dump(target_data, f)

    print(f"Source file saved at: {source_file_path}")
    print(f"Target file saved at: {target_file_path}")

# Example usage
split_nuscenes_data('train')
split_nuscenes_data('val')