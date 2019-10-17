from datetime import datetime, timedelta
import numpy as np
import json
import requests
import os
import glob

from os.path import join, dirname, isfile
base_path = dirname(__file__)

for path in glob.glob('parsed_data/*_text.json'):
    basename = '_'.join(os.path.basename(path).split('.')[0].split('_')[:-1])
    with open(join(base_path, f'parsed_data/{basename}_text.json'), 'r') as f:
        data = json.load(f)

    parsed_data = []

    for date in data:
        parsed_data += data[date]

    with open(join(base_path, f'parsed_data/{basename}_flat.json'), 'w') as f:
        json.dump(parsed_data, f, ensure_ascii=False)
