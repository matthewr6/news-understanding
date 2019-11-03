#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;


# In[54]:


import json

with open('parsed_data/news_flat.json') as f:
    data = json.load(f)

data_new = [[w.lower() for w in line] for line in data]
    
print(data_new)


# In[55]:


id2word = gensim.corpora.Dictionary(data_new)


# In[61]:


corpus = [id2word.doc2bow(text) for text in data_new]
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=10)


# In[62]:


def get_lda_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict)


# In[63]:


import pandas as pd
get_lda_topics(lda, 10)


# In[59]:


num_topics = 10

data_new_sentences = [' '.join(text) for text in data_new]

vectorizer = CountVectorizer(analyzer='word', max_features=5000);
x_counts = vectorizer.fit_transform(data_new_sentences);
transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(x_counts);
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
model = NMF(n_components=num_topics, init='nndsvd');

model.fit(xtfidf_norm)

def get_nmf_topics(model, n_top_words):
    feat_names = vectorizer.get_feature_names()
    word_dict = {};
    for i in range(num_topics):  
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    return pd.DataFrame(word_dict);

get_nmf_topics(model, 20)


# In[60]:


#Topic 1: ???
#Topic 2: General news
#Topic 3: ???
#Topic 4: Democratic debates
#Topic 5: Impeachment
#Topic 6: Brexit/EU
#Topic 7: ???
#Topic 8: Progressives
#Topic 9: Giuliani
#Topic 10: Israel?

