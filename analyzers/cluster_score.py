import pandas as pd
import numpy as np
import sklearn
import sys
# import nltk
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import normalize
# import pickle
import json
import os
import glob
from os.path import join, dirname, isfile
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score

score_func = davies_bouldin_score

# import seaborn as sns

# danger
sys.path.append('.')
from extractors.raw_count import raw_count_topics
from extractors.intracluster_proportion import intracluster_proportion_topics
from extractors.centroid_spread import centroid_spread_topics

topic_extractors = {
    'raw_count': raw_count_topics,
    'intracluster_proportion': intracluster_proportion_topics,
    'centroid_spread': centroid_spread_topics,
}
topic_extractor = None

base_path = dirname(__file__)
model_name = os.path.basename(__file__).split('.')[0]

if len(sys.argv) < 3 or sys.argv[2] not in topic_extractors:
    print('python3 regression.py [model] [topic extractor]')
    print('Model options:')
    for path in glob.glob(join(base_path, '../models/*.py')):
        print(os.path.basename(path).split('.')[0])
    print('\nTopic extractors:')
    for path in glob.glob(join(base_path, '../extractors/*.py')):
        print(os.path.basename(path).split('.')[0])
    # list models
    sys.exit()

model_name = sys.argv[1]
topic_extractor = topic_extractors[sys.argv[2]]

y = []
X = []
with open(join(base_path, f'../models/models/{model_name}/clusters.json'), 'r') as f:
    clusters = json.load(f)

for label in clusters:
    X += [' '.join(l) for l in clusters[label]]
    y += [int(label)] * len(clusters[label])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X).toarray()

def intersect(input_text, topics):
    return [i for i in input_text if i in topics]


print('Original score:', score_func(X, y))

# how relevant is this
topics, cluster_topics = topic_extractor(clusters)
X_keywords = []
for label in clusters:
    X_keywords += [' '.join(intersect(l, topics)) for l in clusters[label]]

X_keywords = vectorizer.transform(X_keywords).toarray()
print('Keyword-only score:', score_func(X_keywords, y))
