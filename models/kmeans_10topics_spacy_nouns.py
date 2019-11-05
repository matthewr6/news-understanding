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

featurizer_name = 'spacy_nouns'
model_name = os.path.basename(__file__).split('.')[0]

# config
num_topics = 10
top_n_words = 25

with open(join(base_path, f'../featurizers/featurized_data/{featurizer_name}.json')) as f:
    data = [[w.lower() for w in line] for line in json.load(f)]

vectorizer = CountVectorizer()

vectorizer.fit([' '.join(d) for d in data])

if False and __name__ == '__main__':
    print('Creating model')
    X = vectorizer.transform([' '.join(d) for d in data])
    model = KMeans(n_clusters=num_topics, max_iter=500).fit(X)
    os.makedirs(join(base_path, f'models/{model_name}'), exist_ok=True)
    with open(join(base_path, f'models/{model_name}/model'), 'wb') as f:
        pickle.dump(model, f)

with open(join(base_path, f'models/{model_name}/model'), 'rb') as f:
    model = pickle.load(f)

def predict(text):
    X = vectorizer.transform(text)
    return model.predict(X)

def get_topics():
    topics = {}
    y = model.predict(vectorizer.transform([' '.join(d) for d in data]))
    for idx, label in enumerate(y):
        if label not in topics:
            topics[label] = collections.defaultdict(int)
        for word in data[idx]:
            topics[label][word] += 1
    topic_words = []
    for label, words in topics.items():
        c = collections.Counter(words)
        topic_words += [w[0] for w in c.most_common(top_n_words)]
    return set(topic_words)

topics = None
if os.path.exists(join(base_path, f'models/{model_name}/topics')):
    with open(join(base_path, f'models/{model_name}/topics'), 'rb') as f:
        topics = pickle.load(f)
else:
    print('Calculating topic keywords...')
    topics = get_topics()
    with open(join(base_path, f'models/{model_name}/topics'), 'wb') as f:
        pickle.dump(topics, f)

