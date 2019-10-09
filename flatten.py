from datetime import datetime, timedelta
import numpy as np
import json
import requests
import os

from os.path import join, dirname, isfile
path = dirname(__file__)

with open(join(path, 'news_text.json'), 'r') as f:
    data = json.load(f)

parsed_data = []

for date in data:
    parsed_data += data[date]

with open(join(path, 'news_flat.json'), 'w') as f:
    json.dump(parsed_data, f, ensure_ascii=False)
