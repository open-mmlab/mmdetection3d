#!/usr/bin/env python
import glob
import re
from os import path as osp

url_prefix = 'https://github.com/open-mmlab/mmdetection3d/blob/master/'

files = sorted(glob.glob('../configs/*/README.md'))

stats = []
titles = []
num_ckpts = 0

for f in files:
    url = osp.dirname(f.replace('../', url_prefix))

    with open(f, 'r') as content_file:
        content = content_file.read()

    title = content.split('\n')[0].replace('#', '')
    ckpts = set(x.lower().strip()
                for x in re.findall(r'https?://download.*\.pth', content)
                if 'mmdetection3d' in x)
    if len(ckpts) == 0:
        continue

    titles.append(title)
    num_ckpts += len(ckpts)
    statsmsg = f"""
\t* [{title}]({url}) ({len(ckpts)} ckpts)
"""
    stats.append((title, ckpts, statsmsg))

msglist = '\n'.join(x for _, _, x in stats)

modelzoo = f"""
\n## Model Zoo Statistics

* Number of papers: {len(set(titles))}
* Number of checkpoints: {num_ckpts}
{msglist}
"""

with open('model_zoo.md', 'a') as f:
    f.write(modelzoo)
