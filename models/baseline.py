import pandas as pd
import numpy as np
import scipy
import sklearn
import sys
from nltk.corpus import stopwords
import nltk
from gensim.models import ldamodel
import gensim.corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import pickle
import json

with open('parsed_data/news_flat.json') as f:
    data = json.load(f)

data_new = [[w.lower() for w in line] for line in data]

id2word = gensim.corpora.Dictionary(data_new)

corpus = [id2word.doc2bow(text) for text in data_new]
num_topics = 10
top_n_words = 25
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

def get_lda_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = top_n_words);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict)

lda_topic_words = []
for i in range(num_topics):
    lda_topic_words.append([w[0] for w in lda.show_topic(i, topn=top_n_words)])
# lda_topic_words = [lda.show_topic(i, topn=top_n_words) for i in range(num_topics)]

def intersect(a, b):
    return list(set(a) & set(b))

import pandas as pd

topics = get_lda_topics(lda, num_topics)
print(topics)

def predict_topic(text):
    doc_bow = id2word.doc2bow(text.lower().split(' '))
    topics = sorted(lda[doc_bow],key=lambda x:x[1],reverse=True)
    return topics[0][0]

data_sentences = [' '.join(text) for text in data_new]

vectorizer = CountVectorizer()
x_counts = vectorizer.fit_transform(data_sentences)
# transformer = TfidfTransformer(smooth_idf=False);
# x_tfidf = transformer.fit_transform(x_counts);
# xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
print(x_counts.shape)

X = x_counts

y = []
for new in data_new:
    y.append(predict_topic(' '.join(new)))


# from sklearn.utils import shuffle
# # X, y = shuffle(x_counts, y)

# split_idx = round(0.8 * len(X))

# X_train, y_train = X[:split_idx], y[:split_idx]
# X_test, y_test = X[split_idx:], y[split_idx:]

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

bayes = MultinomialNB()
lr = LogisticRegression()

bayes_score = cross_val_score(bayes, X, y, cv=5, scoring='accuracy')
lr_score = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
print('not keywords', np.mean(bayes_score), np.mean(lr_score))


# ### AAAAA

stripped_data = [' '.join(intersect(new, lda_topic_words[y[idx]])) for idx, text in enumerate(data_new)]

vectorizer = CountVectorizer()
x_counts = vectorizer.fit_transform(stripped_data)
# # transformer = TfidfTransformer(smooth_idf=False);
# # x_tfidf = transformer.fit_transform(x_counts);
# # xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
# print(x_counts.shape)

X = x_counts

from sklearn.utils import shuffle
# X, y = shuffle(x_counts, y)

# split_idx = round(0.8 * len(X))

# X_train, y_train = X[:split_idx], y[:split_idx]
# X_test, y_test = X[split_idx:], y[split_idx:]

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

bayes = MultinomialNB()
lr = LogisticRegression()

bayes_score = cross_val_score(bayes, X, y, cv=5, scoring='accuracy')
lr_score = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
print('keywords', np.mean(bayes_score), np.mean(lr_score))

# model = NMF(n_components=num_topics, init='nndsvd');

# model.fit(xtfidf_norm)

# def get_nmf_topics(model, n_top_words):
#     feat_names = vectorizer.get_feature_names()
#     word_dict = {};
#     for i in range(num_topics):  
#         words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1]
#         words = [feat_names[key] for key in words_ids]
#         word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
#     return pd.DataFrame(word_dict);

# topics = get_nmf_topics(model, top_n_words)

# print(topics)
