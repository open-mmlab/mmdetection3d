"""Script to gather benchmarked models and prepare them for upload.

Usage:
python gather_models.py ${root_path} ${out_dir}
"""

import argparse
import glob
import json
import mmcv
import shutil
import subprocess
import torch
from os import path as osp

# build schedule look-up table to automatically find the final model
SCHEDULES_LUT = {
    '_1x_': 12,
    '_2x_': 24,
    '_20e_': 20,
    '_3x_': 36,
    '_4x_': 48,
    '_24e_': 24,
    '_6x_': 73
}

# TODO: add support for lyft dataset
RESULTS_LUT = {
    'coco': ['bbox_mAP', 'segm_mAP'],
    'nus': ['pts_bbox_NuScenes/NDS', 'NDS'],
    'kitti-3d-3class': [
        'KITTI/Overall_3D_moderate',
        'Overall_3D_moderate',
    ],
    'kitti-3d-car': ['KITTI/Car_3D_moderate_strict', 'Car_3D_moderate_strict'],
    'lyft': ['score'],
    'scannet': ['mAR_0.50'],
    'sunrgbd': ['mAR_0.50']
}


def get_model_dataset(log_json_path):
    for key in RESULTS_LUT:
        if log_json_path.find(key) != -1:
            return key


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', out_file, final_file])
    return final_file


def get_final_epoch(config):
    if config.find('grid_rcnn') != -1 and config.find('2x') != -1:
        # grid_rcnn 2x trains 25 epochs
        return 25

    for schedule_name, epoch_num in SCHEDULES_LUT.items():
        if config.find(schedule_name) != -1:
            return epoch_num


def get_best_results(log_json_path):
    dataset = get_model_dataset(log_json_path)
    max_dict = dict()
    max_memory = 0
    with open(log_json_path, 'r') as f:
        for line in f.readlines():
            log_line = json.loads(line)
            if 'mode' not in log_line.keys():
                continue

            # record memory and find best results & epochs
            if log_line['mode'] == 'train' \
                    and max_memory <= log_line['memory']:
                max_memory = log_line['memory']

            elif log_line['mode'] == 'val':
                result_dict = {
                    key: log_line[key]
                    for key in RESULTS_LUT[dataset] if key in log_line
                }
                if len(max_dict) == 0:
                    max_dict = result_dict
                    max_dict['epoch'] = log_line['epoch']
                elif all(
                    [max_dict[key] <= result_dict[key]
                     for key in result_dict]):
                    max_dict.update(result_dict)
                    max_dict['epoch'] = log_line['epoch']

        max_dict['memory'] = max_memory
        return max_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Gather benchmarked models')
    parser.add_argument(
        'root',
        type=str,
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        'out', type=str, help='output path of gathered models to be stored')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    models_root = args.root
    models_out = args.out
    mmcv.mkdir_or_exist(models_out)

    # find all models in the root directory to be gathered
    raw_configs = list(mmcv.scandir('./configs', '.py', recursive=True))

    # filter configs that is not trained in the experiments dir
    used_configs = []
    for raw_config in raw_configs:
        if osp.exists(osp.join(models_root, raw_config)):
            used_configs.append(raw_config)
    print(f'Find {len(used_configs)} models to be gathered')

    # find final_ckpt and log file for trained each config
    # and parse the best performance
    model_infos = []
    for used_config in used_configs:
        exp_dir = osp.join(models_root, used_config)

        # get logs
        log_json_path = glob.glob(osp.join(exp_dir, '*.log.json'))[0]
        log_txt_path = glob.glob(osp.join(exp_dir, '*.log'))[0]
        model_performance = get_best_results(log_json_path)
        final_epoch = model_performance['epoch']
        final_model = 'epoch_{}.pth'.format(final_epoch)
        model_path = osp.join(exp_dir, final_model)

        # skip if the model is still training
        if not osp.exists(model_path):
            print(f'Expected {model_path} does not exist!')
            continue

        if model_performance is None:
            print(f'Obtained no performance for model {used_config}')
            continue

        model_time = osp.split(log_txt_path)[-1].split('.')[0]
        model_infos.append(
            dict(
                config=used_config,
                results=model_performance,
                epochs=final_epoch,
                model_time=model_time,
                log_json_path=osp.split(log_json_path)[-1]))

    # publish model for each checkpoint
    publish_model_infos = []
    for model in model_infos:
        model_publish_dir = osp.join(models_out, model['config'].rstrip('.py'))
        mmcv.mkdir_or_exist(model_publish_dir)

        model_name = model['config'].split('/')[-1].rstrip(
            '.py') + '_' + model['model_time']
        publish_model_path = osp.join(model_publish_dir, model_name)
        trained_model_path = osp.join(models_root, model['config'],
                                      'epoch_{}.pth'.format(model['epochs']))

        # convert model
        final_model_path = process_checkpoint(trained_model_path,
                                              publish_model_path)

        # copy log
        shutil.copy(
            osp.join(models_root, model['config'], model['log_json_path']),
            osp.join(model_publish_dir, f'{model_name}.log.json'))
        shutil.copy(
            osp.join(models_root, model['config'],
                     model['log_json_path'].rstrip('.json')),
            osp.join(model_publish_dir, f'{model_name}.log'))

        # copy config to guarantee reproducibility
        config_path = model['config']
        config_path = osp.join(
            'configs',
            config_path) if 'configs' not in config_path else config_path
        target_cconfig_path = osp.split(config_path)[-1]
        shutil.copy(config_path,
                    osp.join(model_publish_dir, target_cconfig_path))

        model['model_path'] = final_model_path
        publish_model_infos.append(model)

    models = dict(models=publish_model_infos)
    print(f'Totally gathered {len(publish_model_infos)} models')
    mmcv.dump(models, osp.join(models_out, 'model_info.json'))


if __name__ == '__main__':
    main()
