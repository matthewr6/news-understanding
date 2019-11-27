import sys
from sklearn.cluster import KMeans
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import json
import os
import collections
import numpy as np
from os.path import join, dirname, isfile
base_path = dirname(__file__)

featurizer_name = 'median_word_length'
model_name = os.path.basename(__file__).split('.')[0]

# config
num_topics = 5
top_n_words = 25

with open(join(base_path, f'../featurizers/featurized_data/{featurizer_name}.json')) as f:
    data = [[w.lower() for w in line] for line in json.load(f)]

vectorizer = CountVectorizer()

vectorizer.fit([' '.join(d) for d in data])

if __name__ == '__main__':
    print('Creating model')
    X = vectorizer.transform([' '.join(d) for d in data])
    model = KMeans(n_clusters=num_topics, max_iter=500).fit(X)
    os.makedirs(join(base_path, f'models/{model_name}'), exist_ok=True)
    with open(join(base_path, f'models/{model_name}/model'), 'wb') as f:
        pickle.dump(model, f)
    y = model.predict(vectorizer.transform([' '.join(d) for d in data]))
    groupings = {}
    for i in np.unique(y):
        groupings[str(i)] = []
    for idx, d in enumerate(data):
        groupings[str(y[idx])].append(d)
    with open(join(base_path, f'models/{model_name}/clusters.json'), 'w') as f:
        json.dump(groupings, f, indent=4)

with open(join(base_path, f'models/{model_name}/model'), 'rb') as f:
    model = pickle.load(f)

def predict(text):
    X = vectorizer.transform(text)
    return model.predict(X)
