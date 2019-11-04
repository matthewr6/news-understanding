import sys
from sklearn.cluster import KMeans
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import json
import os
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
    # print(set(KMeans(eps=0.01).fit_predict(X)))

with open(join(base_path, f'models/{model_name}/model'), 'rb') as f:
    model = pickle.load(f)

def predict(text):
    X = vectorizer.transform(text)
    return model.predict(X)

# def get_topics():
#     lda_topic_words = []
#     for i in range(num_topics):
#         # lda_topic_words.append([w[0] for w in lda.show_topic(i, topn=top_n_words)])
#         lda_topic_words += [w[0] for w in lda.show_topic(i, topn=top_n_words)]
#     # return lda_topic_words
#     return set(lda_topic_words)

# topics = get_topics()
