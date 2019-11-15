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

with open(join(base_path, '../datasets/data.json'), 'r') as f:
    data = json.load(f)

pattern = re.compile('([^\s\w]|_)+')

stripped_data = []
for text in data:
    stripped_data += pattern.sub('', text).strip().lower().split(' ')

lengths = [len(w) for w in stripped_data]
print('median =', np.median(lengths))
print('mean =', np.mean(lengths))

# import matplotlib.pyplot as plt

# plt.hist(lengths)
# plt.show()

MIN_LENGTH = np.median(lengths)

def parse_article(text):
    stripped_text = pattern.sub('', text).strip().lower().split(' ')
    return [t for t in stripped_text if len(t) > MIN_LENGTH]

with open(join(base_path, f'featurized_data/{featurizer_name}.json'), 'w') as f:
    json.dump([parse_article(d) for d in data], f, ensure_ascii=False)
