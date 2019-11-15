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

def intersect(input_text, topics):
    return [i for i in input_text if i in topics]
    # return list(set(a) & set(b))

weighted = True
print(f'weighted = {weighted}')
stripped_data = [' '.join(intersect(text.split(' '), model.weighted_topics if weighted else model.topics)) for idx, text in enumerate(data_sentences)]
# stripped_weighted_data = [' '.join(intersect(text.split(' '), model.weighted_topics)) for idx, text in enumerate(data_sentences)]

# use weighted topics instead so stuff like trump is less relevant?

# get featurized data
vectorizer = CountVectorizer()
X_base = vectorizer.fit_transform(data_sentences)
X_keywords = vectorizer.fit_transform(stripped_data)
# X_weighted_keywords = vectorizer.fit_transform(stripped_weighted_data)


# X_base, X_keywords, X_weighted_keywords, y = sklearn.utils.shuffle(X_base, X_keywords, X_weighted_keywords, y)

iters = 10
# set up classifiers
lr = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)

# predict
base_scores = np.array([])
keywords_scores = np.array([])
print('Beginning cross validation on logistic regression...')
for i in range(iters):
    X_base, X_keywords, y = sklearn.utils.shuffle(X_base, X_keywords, y)
    base_scores = np.concatenate([base_scores, cross_val_score(lr, X_base, y, cv=5, scoring='accuracy')])
    keywords_scores = np.concatenate([keywords_scores, cross_val_score(lr, X_keywords, y, cv=5, scoring='accuracy')])
    print(f'    Completed iter {i + 1}/{iters}')
    # print('    featurized data', np.mean(base_scores))
    # print('    keywords only', np.mean(keywords_scores))
    # print('    overall score - keywords score', np.mean(base_scores) - np.mean(keywords_scores))

print('featurized data', np.mean(base_scores))
print('keywords only', np.mean(keywords_scores))
print('overall score - keywords score', np.mean(base_scores) - np.mean(keywords_scores))

print('terms')
# get terms on a per topic basis
# print(model.topics)
