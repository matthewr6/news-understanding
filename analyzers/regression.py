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
y = []
for d in data_sentences:
    y.append(model.predict(d))

def intersect(a, b):
    return list(set(a) & set(b))

stripped_data = [' '.join(intersect(text.split(' '), model.topics)) for idx, text in enumerate(data_sentences)]

# get featurized data
vectorizer = CountVectorizer()
X_base = vectorizer.fit_transform(data_sentences)
X_keywords = vectorizer.fit_transform(stripped_data)

# set up classifiers
bayes = MultinomialNB()
lr = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)

# predict
bayes_score = cross_val_score(bayes, X_base, y, cv=5, scoring='accuracy')
lr_score = cross_val_score(lr, X_base, y, cv=5, scoring='accuracy')
print('featurized data', np.mean(bayes_score), np.mean(lr_score))
bayes_score = cross_val_score(bayes, X_keywords, y, cv=5, scoring='accuracy')
lr_score = cross_val_score(lr, X_keywords, y, cv=5, scoring='accuracy')
print('keywords only', np.mean(bayes_score), np.mean(lr_score))
