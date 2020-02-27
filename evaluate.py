# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:28:50 2020

@author: toikkal
"""

import imageio
import math
import numpy as np
import os
from pathlib import Path


cwd = Path('//ad.tuni.fi/home/toikkal/StudentDocuments/Documents/media-analysis')
metrics = []
file_changes = []
image = 0
img_indices = range(1, 23)
for index in img_indices:
    img_index = index
    filename = cwd / 'data' / f'kodim{img_index}.png'
    result_filename = cwd / 'results' / f'result-{img_index-1}.png'
    try:
        img_data = imageio.imread(filename)
        img_size = os.stat(filename).st_size
        image += 1
    except Exception:
        print("Could not read original file, skipping...")
        continue
    result_image = imageio.imread(result_filename)
    result_size = os.stat(result_image).st_size
    mse = np.mean((img_data-result_image)**2)
    if mse == 0:
        d=100
    else:
        d = 20 * math.log10(255/math.sqrt(mse))
    metrics.append(d)
    file_changes.append(img_size-result_size)
print(np.mean(metrics))
print(np.mean(file_changes))