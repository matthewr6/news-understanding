from datetime import datetime, timedelta
import numpy as np
import json
import requests
import os
import glob

from os.path import join, dirname, isfile
base_path = dirname(__file__)

all_data = {}

for path in glob.glob(join(base_path, 'raw_data/*.json')):
    with open(path, 'r') as f:
        data = json.load(f)

    for date in data:
        if date not in all_data:
            all_data[date] = []
        all_data[date] += data[date]['articles']
        all_data[date] = [i for n, i in enumerate(all_data[date]) if i not in all_data[date][n + 1:]]

with open(join(base_path, 'all_data.json'), 'w') as f:
    json.dump(all_data, f, ensure_ascii=False)
