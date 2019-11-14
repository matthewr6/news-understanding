import numpy as np
import json
import requests
import os
import glob
import regex as re

import nltk
from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english')) 

featurizer_name = os.path.basename(__file__).split('.')[0]

from os.path import join, dirname, isfile
base_path = dirname(__file__)

table = str.maketrans('', '', string.punctuation)

# parameter
N = 2

def ngram(terms, n=2):
    ret = []
    for i in range(len(terms) - (n - 1)):
        built = ''
        for j in range(n):
            built += terms[i + j] + ' '
        ret.append(built[:-1])
    return ret

pattern = re.compile('([^\s\w]|_)+')

def parse_article(text):
    stripped_text = pattern.sub('', text).strip().lower().split(' ')
    base = [t for t in stripped_text if t not in stop_words]
    return ngram(base, n=N)

with open(join(base_path, '../datasets/data.json'), 'r') as f:
    data = json.load(f)

with open(join(base_path, f'featurized_data/{featurizer_name}.json'), 'w') as f:
    json.dump([parse_article(d) for d in data], f, ensure_ascii=False)
