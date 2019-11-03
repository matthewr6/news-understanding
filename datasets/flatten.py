from datetime import datetime, timedelta
import numpy as np
import json
import requests
import os
import glob

from os.path import join, dirname, isfile
base_path = dirname(__file__)

with open(join(base_path, 'all_data.json'), 'r') as f:
    data = json.load(f)

parsed_data = []

def get_text(article):
    text = ''
    if 'title' in article and article['title']:
        text = article['title']
    if 'description' in article and article['description']:
        text += ' ' + article['description']
    return text.strip()

for date in data:
    # get the title + description
    parsed_data += [get_text(d) for d in data[date]]

with open(join(base_path, 'data.json'), 'w') as f:
    json.dump(parsed_data, f, ensure_ascii=False)
