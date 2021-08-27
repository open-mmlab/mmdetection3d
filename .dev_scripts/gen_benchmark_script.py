import argparse
import re
from os import path as osp


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate benchmark training/testing scripts')
    parser.add_argument(
        '--input_file',
        required=False,
        type=str,
        help='Input file containing the paths '
        'of configs to be trained/tested.')
    parser.add_argument(
        '--output_file',
        required=True,
        type=str,
        help='Output file containing the '
        'commands to train/test selected models.')
    parser.add_argument(
        '--gpus_per_node',
        type=int,
        default=8,
        help='GPUs per node config for slurm, '
        'should be set according to your slurm environment')
    parser.add_argument(
        '--cpus_per_task',
        type=int,
        default=5,
        help='CPUs per task config for slurm, '
        'should be set according to your slurm environment')
    parser.add_argument(
        '--mode', type=str, default='train', help='Train or test')
    parser.add_argument(
        '--long_work_dir',
        action='store_true',
        help='Whether use full relative path of config as work dir')
    parser.add_argument(
        '--max_keep_ckpts',
        type=int,
        default=1,
        help='The max number of checkpoints saved in training')
    parser.add_argument(
        '--full_log',
        action='store_true',
        help='Whether save full log in a file')

    args = parser.parse_args()
    return args


args = parse_args()
assert args.mode in ['train', 'test'], 'Currently we only support ' \
    'automatically generating training or testing scripts.'

config_paths = []

if args.input_file is not None:
    with open(args.input_file, 'r') as fi:
        config_paths = fi.read().strip().split('\n')
else:
    while True:
        print('Please type a config path and '
              'press enter (press enter directly to exit):')
        config_path = input()
        if config_path != '':
            config_paths.append(config_path)
        else:
            break

script = '''PARTITION=$1
CHECKPOINT_DIR=$2

'''

for i, config_path in enumerate(config_paths):
    root_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    if not osp.exists(osp.join(root_dir, config_path)):
        print(f'Invalid config path (does not exist):\n{config_path}')
        continue

    config_name = config_path.split('/')[-1][:-3]
    match_obj = re.match(r'^.*_[0-9]+x([0-9]+)_.*$', config_name)
    if match_obj is None:
        print(f'Invalid config path (no GPU num in '
              f'config name):\n{config_path}')
        continue

    gpu_num = int(match_obj.group(1))
    work_dir_name = config_path if args.long_work_dir else config_name

    script += f"echo '{config_path}' &\n"
    if args.full_log:
        script += f'mkdir -p $CHECKPOINT_DIR/{work_dir_name}\n'

    # training commands
    script += f'GPUS={gpu_num} GPUS_PER_NODE={args.gpus_per_node} ' \
              f'CPUS_PER_TASK={args.cpus_per_task} ' \
              f'./tools/slurm_train.sh $PARTITION {config_name} ' \
              f'{config_path} \\\n'
    script += f'$CHECKPOINT_DIR/{work_dir_name} --cfg-options ' \
              f'checkpoint_config.max_keep_ckpts={args.max_keep_ckpts} \\\n' \

    # if output full log, redirect stdout and stderr to
    # another log file in work dir
    if args.full_log:
        script += f'2>&1|tee $CHECKPOINT_DIR/{work_dir_name}/FULL_LOG.txt &\n'
    else:
        script += '>/dev/null &\n'

    if i != len(config_paths) - 1:
        script += '\n'

    print(f'Successfully generated script for {config_name}')

with open(args.output_file, 'w') as fo:
    fo.write(script)
