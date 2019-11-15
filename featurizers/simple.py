import numpy as np
import json
import requests
import os
import glob
import regex as re

import string

featurizer_name = os.path.basename(__file__).split('.')[0]

from os.path import join, dirname, isfile
base_path = dirname(__file__)

pattern = re.compile('([^\s\w]|_)+')

def parse_article(text):
    return pattern.sub('', text).strip().lower().split(' ')

with open(join(base_path, '../datasets/data.json'), 'r') as f:
    data = json.load(f)

with open(join(base_path, f'featurized_data/{featurizer_name}.json'), 'w') as f:
    json.dump([parse_article(d) for d in data], f, ensure_ascii=False)
