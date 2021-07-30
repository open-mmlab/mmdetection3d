import argparse
import re


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
        '--mode', type=str, default='train', help='Train or test')

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
    config_name = config_path.split('/')[-1][:-3]
    match_obj = re.match(r'.*_[0-9]+x([0-9]+)_.*', config_name)

    gpu_num = int(match_obj.group(1))
    script += f"echo '{config_path}' &\n"
    script += f'GPUS={gpu_num}  GPUS_PER_NODE={gpu_num}  CPUS_PER_TASK=5 ' \
              f'./tools/slurm_train.sh $PARTITION {config_name} ' \
              f'{config_path} \\\n'
    script += f'$CHECKPOINT_DIR/{config_name} --cfg-options ' \
              f'checkpoint_config.max_keep_ckpts=1 >/dev/null &\n'
    if i != len(config_paths) - 1:
        script += '\n'

with open(args.output_file, 'w') as fo:
    fo.write(script)
