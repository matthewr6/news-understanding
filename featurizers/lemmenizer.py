import numpy as np
import json
import requests
import os
import glob
import regex as re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer= PorterStemmer()

import spacy
nlp = spacy.load("en_core_web_sm")

featurizer_name = os.path.basename(__file__).split('.')[0]

from os.path import join, dirname, isfile
base_path = dirname(__file__)

def parse_article(text):
    doc = nlp(text.strip())
    tokens = [stemmer.stem(token.lemma_) for token in doc if 'NN' in token.tag_]
    return tokens

with open(join(base_path, '../datasets/data.json'), 'r') as f:
    data = json.load(f)

with open(join(base_path, f'featurized_data/{featurizer_name}.json'), 'w') as f:
    json.dump([parse_article(d) for d in data], f, ensure_ascii=False)