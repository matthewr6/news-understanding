from datetime import datetime, timedelta
import numpy as np
import json
import requests
import os

from os.path import join, dirname, isfile
path = dirname(__file__)

with open(join(path, 'raw_data/news.json'), 'r') as f:
    data = json.load(f)

total = 0
count = 0
for k in data:
    total += len(data[k]['articles'])
    count += 1
    print(k, len(data[k]['articles']))

print('Total:', total)

print('Avg:', total / count)

print('')

# with open(join(path, 'news_flat.json'), 'r') as f:
#     print(len(json.load(f)))
