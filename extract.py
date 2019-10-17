from datetime import datetime, timedelta
import numpy as np
import json
import requests
import os
import glob
import regex as re

import spacy
nlp = spacy.load("en_core_web_sm")


from os.path import join, dirname, isfile
base_path = dirname(__file__)

def should_include_word(word):
    return len(word) >= min_len


def parse_article(article):
    text = ''
    if 'title' in article and article['title']:
            text = article['title']
    if 'description' in article and article['description']:
            text += ' ' + article['description']
    doc = nlp(text.strip())
    tokens = [token.lemma_ for token in doc if 'NN' in token.tag_]
    return tokens

for path in glob.glob('raw_data/*.json'):
    basename = os.path.basename(path).split('.')[0]

    with open(join(base_path, f'raw_data/{basename}.json'), 'r') as f:
        data = json.load(f)

    parsed_data = {}
    for date in data:
        articles = data[date]['articles']
        parsed_data[date] = [parse_article(a) for a in articles]

    with open(join(base_path, f'parsed_data/{basename}_text.json'), 'w') as f:
        json.dump(parsed_data, f, ensure_ascii=False)

    print(basename)

# do a PCA or some other sort of dimensionality reduction?
# (e.g. simple way - if the words are together in > N% of samples, combine them)