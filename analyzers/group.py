"""
Script to create output of format:
{
    label: {
        keywords: [...],
        weighted_keywords: [],
        articles: [...]
    },
    ...
}
"""

import pandas as pd
import numpy as np
import sklearn
import sys
# import nltk
import gensim.corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.preprocessing import normalize
# import pickle
import json
import os
import glob
from os.path import join, dirname, isfile
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

base_path = dirname(__file__)
model_name = os.path.basename(__file__).split('.')[0]

if len(sys.argv) < 2:
    print('python3 regression.py [model]')
    print('Model options:')
    for path in glob.glob(join(base_path, '../models/*.py')):
        print(os.path.basename(path).split('.')[0])
    # list models
    sys.exit()

# models are dependent on featurizers
model_name = sys.argv[1]
print(f'Analyzing {model_name}')

# import stuff from correct file
# specifically featurizer_name, predict, and topics
import importlib.util
model_path = join(base_path, f'../models/{model_name}.py')
spec = importlib.util.spec_from_file_location("model", model_path)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

# get featurized data + keyword only featurized data
data_sentences = [' '.join(text).lower() for text in model.data]

# get true labels
y = model.predict(data_sentences)

groupings = {}
for label in np.unique(y):
    groupings[str(label)] = {
        'keywords': list(model.topic_keywords[label]),
        'weighted_keywords': list(model.weighted_topic_keywords[label]),
        'articles': [],
    }

# add keywords...
 
for idx, sentence in enumerate(data_sentences):
    label = y[idx]
    groupings[str(label)]['articles'].append(sentence)

with open(join(base_path, f'groupings/{model_name}.json'), 'w') as f:
    json.dump(groupings, f, indent=4)
