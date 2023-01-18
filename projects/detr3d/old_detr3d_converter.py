from argparse import ArgumentParser

import torch

parser = ArgumentParser()
parser.add_argument('src', default='old.pth')
parser.add_argument('dst', default='new.pth')  # ('training','validation')
parser.add_argument('--code_size', type=int, default='10')
args = parser.parse_args()
model = torch.load(args.src)
code_size = args.code_size
if model['meta'].get('detr3d_convert_tag') is not None:
    print('this model has already converted!')
else:
    print('converting...')
    # (cx, cy, w, l, cz, h, sin(φ), cos(φ), vx, vy)
    for key in model['state_dict']:
        tsr = model['state_dict'][key]
        if 'reg_branches' in key and tsr.shape[0] == code_size:
            print(key, ' with ', tsr.shape, 'has changed')
            tsr[[2, 3], ...] = tsr[[3, 2], ...]
            tsr[[6, 7], ...] = -tsr[[7, 6], ...]
    model['meta']['detr3d_convert_tag'] = True
    torch.save(model, args.dst)
    print('done...')
