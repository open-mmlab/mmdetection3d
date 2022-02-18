#!/usr/bin/env python
import functools as func
import glob
import re
from os import path as osp

import numpy as np

url_prefix = 'https://github.com/open-mmlab/mmdetection3d/blob/master/'

files = sorted(glob.glob('../configs/*/README.md'))

stats = []
titles = []
num_ckpts = 0

for f in files:
    url = osp.dirname(f.replace('../', url_prefix))

    with open(f, 'r') as content_file:
        content = content_file.read()

    title = content.split('\n')[0].replace('#', '').strip()
    ckpts = set(x.lower().strip()
                for x in re.findall(r'https?://download.*\.pth', content)
                if 'mmdetection3d' in x)
    if len(ckpts) == 0:
        continue

    _papertype = [x for x in re.findall(r'<!-- \[([A-Z]+)\] -->', content)]
    assert len(_papertype) > 0
    papertype = _papertype[0]

    paper = set([(papertype, title)])

    titles.append(title)
    num_ckpts += len(ckpts)
    statsmsg = f"""
\t* [{papertype}] [{title}]({url}) ({len(ckpts)} ckpts)
"""
    stats.append((paper, ckpts, statsmsg))

allpapers = func.reduce(lambda a, b: a.union(b), [p for p, _, _ in stats])
msglist = '\n'.join(x for _, _, x in stats)

papertypes, papercounts = np.unique([t for t, _ in allpapers],
                                    return_counts=True)
countstr = '\n'.join(
    [f'   - {t}: {c}' for t, c in zip(papertypes, papercounts)])

modelzoo = f"""
\n## Model Zoo Statistics

* Number of papers: {len(set(titles))}
{countstr}

* Number of checkpoints: {num_ckpts}
{msglist}
"""

with open('model_zoo.md', 'a') as f:
    f.write(modelzoo)
