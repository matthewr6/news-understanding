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

featurizer_name = 'ngram_2'
model_name = os.path.basename(__file__).split('.')[0]

# config
num_topics = 10
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
    topic_keywords = {}
    for label, words in topics.items():
        if label not in topic_keywords:
            topic_keywords[label] = set()
        c = collections.Counter(words)
        # topic_words += [w[0] for w in c.most_common(top_n_words)]
        for w in c.most_common(top_n_words):
            topic_words.append(w[0])
            topic_keywords[label].add(w[0])
    return (set(topic_words), topic_keywords)

def normalize_dict(d):
    total = 0
    for k, v in d.items():
        total += v
    for k in d:
        d[k] /= total
    return d

asdf = 100

# could multiply by inverse of total?
def get_weighted_topics():
    topics = {}
    y = model.predict(vectorizer.transform([' '.join(d) for d in data]))
    for idx, label in enumerate(y):
        if label not in topics:
            topics[label] = collections.defaultdict(int)
        for word in data[idx]:
            topics[label][word] += 1
    normalized_terms = {}
    totals = collections.defaultdict(float)
    for label, words in topics.items():
        c = collections.Counter(words)
        normalized_terms[label] = normalize_dict(dict(c.most_common(asdf)))
        for term, proportion in normalized_terms[label].items():
            totals[term] += proportion
    for label, terms in normalized_terms.items():
        for term, value in terms.items():
            normalized_terms[label][term] *= value / totals[term]
    final_terms = []
    topic_keywords = {}
    for label, words in normalized_terms.items():
        if label not in topic_keywords:
            topic_keywords[label] = set()
        c = collections.Counter(words)
        for w in c.most_common(top_n_words):
            final_terms.append(w[0])
            topic_keywords[label].add(w[0])
    return (set(final_terms), topic_keywords)

topics = None
if os.path.exists(join(base_path, f'models/{model_name}/topics')) and __name__ != '__main__':
    with open(join(base_path, f'models/{model_name}/topics'), 'rb') as f:
        topics, topic_keywords = pickle.load(f)
else:
    print('Calculating topic keywords...')
    topics, topic_keywords = get_topics()
    with open(join(base_path, f'models/{model_name}/topics'), 'wb') as f:
        pickle.dump((topics, topic_keywords), f)

weighted_topics = None
if os.path.exists(join(base_path, f'models/{model_name}/weighted_topics')) and __name__ != '__main__':
    with open(join(base_path, f'models/{model_name}/weighted_topics'), 'rb') as f:
        weighted_topics, weighted_topic_keywords = pickle.load(f)
else:
    print('Calculating topic weighted keywords...')
    weighted_topics, weighted_topic_keywords = get_weighted_topics()
    with open(join(base_path, f'models/{model_name}/weighted_topics'), 'wb') as f:
        pickle.dump((weighted_topics, weighted_topic_keywords), f)
