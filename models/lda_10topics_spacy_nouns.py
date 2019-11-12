import sys
from gensim.models.ldamodel import LdaModel
import gensim.corpora
import json
import os
from os.path import join, dirname, isfile
base_path = dirname(__file__)

featurizer_name = 'spacy_nouns'
model_name = os.path.basename(__file__).split('.')[0]

# config
num_topics = 10
top_n_words = 25

with open(join(base_path, f'../featurizers/featurized_data/{featurizer_name}.json')) as f:
    data = [[w.lower() for w in line] for line in json.load(f)]

id2word = gensim.corpora.Dictionary(data)
corpus = [id2word.doc2bow(text) for text in data]

if __name__ == '__main__':
    print('Creating model')
    lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
    os.makedirs(join(base_path, f'models/{model_name}'), exist_ok=True)
    lda.save(join(base_path, f'models/{model_name}/model'))

lda = LdaModel.load(join(base_path, f'models/{model_name}/model'))

def predict(texts):
    ret = []
    for text in texts:
        doc_bow = id2word.doc2bow(text.lower().split(' '))
        topics = sorted(lda[doc_bow], key=lambda x:x[1],reverse=True)
        ret.append(topics[0][0])
    return ret

def get_topics():
    lda_topic_words = []
    for i in range(num_topics):
        lda_topic_words += [w[0] for w in lda.show_topic(i, topn=top_n_words)]
    return set(lda_topic_words)

def get_weighted_topics():
    lda_topic_words = []
    for i in range(num_topics):
        topic_keywords = lda.show_topic(i, topn=top_n_words)
        lda_topic_words += [w[0] for w in lda.show_topic(i, topn=top_n_words)]
    return set(lda_topic_words)

topics = get_topics()

# get_weighted_topics()
