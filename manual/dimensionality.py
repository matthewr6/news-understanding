import json
from os.path import join, dirname, isfile
base_path = dirname(__file__)

for which in ['simple', 'no_stopwords', 'spacy_nouns', 'median_word_length']:
    with open(join(base_path, f'../featurizers/featurized_data/{which}.json'), 'r') as f:
        data = json.load(f)
    d = []
    for i in data:
        d += i
    print(f'|{which}| = {len(set(d))}')