import sys
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import json
import os
import collections
import numpy as np
from os.path import join, dirname, isfile
#danger
sys.path.append('.')
from vendor.tkm import TKMCore, algTools
base_path = dirname(__file__)

featurizer_name = 'spacy_nouns'
model_name = os.path.basename(__file__).split('.')[0]

# config
num_topics = 10

with open(join(base_path, f'../featurizers/featurized_data/{featurizer_name}.json')) as f:
    data = [[w.lower() for w in line] for line in json.load(f)]

m_docs, id2word = algTools.process_corpus([' '.join(line) for line in data])

if __name__ == '__main__':
    print('Creating model')
    
    model = TKMCore.TKMCore(
        m_docs=m_docs,
        n_words=len(id2word),
        n_topics=num_topics,
        winwid=7,
        alpha=7,
        beta=0.08
    )
    model.run(convergence_constant=0.0001)

    os.makedirs(join(base_path, f'models/{model_name}'), exist_ok=True)
    with open(join(base_path, f'models/{model_name}/model'), 'wb') as f:
        pickle.dump(model, f)

    y, _ = model.get_topics(m_docs)
    y = np.argmax(y, axis=1)
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
    m_docs, id2word = algTools.process_corpus(text)
    y, _ = model.get_topics(m_docs)
    return np.argmax(y, axis=1)
