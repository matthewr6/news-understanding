import sys
from gensim.models.ldamodel import LdaModel
import gensim.corpora
import json
import os
import numpy as np
from os.path import join, dirname, isfile
base_path = dirname(__file__)

featurizer_name = 'no_stopwords'
model_name = os.path.basename(__file__).split('.')[0]

# config
num_topics = 10

with open(join(base_path, f'../featurizers/featurized_data/{featurizer_name}.json')) as f:
    data = [[w.lower() for w in line] for line in json.load(f)]

id2word = gensim.corpora.Dictionary(data)
corpus = [id2word.doc2bow(text) for text in data]

if __name__ == '__main__':
    print('Creating model')
    lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
    os.makedirs(join(base_path, f'models/{model_name}'), exist_ok=True)
    lda.save(join(base_path, f'models/{model_name}/model'))

    y = []
    for text in data:
        doc_bow = id2word.doc2bow(text)
        topics = sorted(lda[doc_bow], key=lambda x:x[1],reverse=True)
        y.append(topics[0][0])
    groupings = {}
    for i in np.unique(y):
        groupings[str(i)] = []
    for idx, d in enumerate(data):
        groupings[str(y[idx])].append(d)
    with open(join(base_path, f'models/{model_name}/clusters.json'), 'w') as f:
        json.dump(groupings, f, indent=4)

lda = LdaModel.load(join(base_path, f'models/{model_name}/model'))

def predict(texts):
    ret = []
    for text in texts:
        doc_bow = id2word.doc2bow(text.lower().split(' '))
        topics = sorted(lda[doc_bow], key=lambda x:x[1],reverse=True)
        ret.append(topics[0][0])
    return ret
