import sys
import json
import os
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms
import numpy as np
from os.path import join, dirname, isfile
from biterm.cbtm import oBTM
# from biterm.btm import oBTM

base_path = dirname(__file__)

featurizer_name = 'spacy_nouns'
model_name = os.path.basename(__file__).split('.')[0]

num_topics = 10
top_n_words = 25

with open(join(base_path, f'../featurizers/featurized_data/{featurizer_name}.json')) as f:
    data = [" ".join([w.lower() for w in line]) for line in json.load(f)]

#topics 
#examples x value_for_each_topics[ topics.argmax() ]

vec = CountVectorizer(stop_words='english')
X = vec.fit_transform(data).toarray()
biterms = vec_to_biterms(X)

# train model
if __name__ == '__main__':
    print("Creating model")
    # vectorize texts
    
    # get vocabulary
    vocab = np.array(vec.get_feature_names())
    
    # get biterms
    biterms = vec_to_biterms(X)
    
    # create btm
    btm = oBTM(num_topics=num_topics, V=vocab)

    chunk_size = 1000
    for i in range(0, len(biterms), chunk_size):
        biterms_chunk = biterms[i:i + chunk_size]
        btm.fit(biterms_chunk, iterations=100)
        print(f'{chunk_size * (i + chunk_size) / len(biterms)}%')
    # topics = btm.fit_transform(biterms, iterations=20)

    #save model to file
    os.makedirs(join(base_path, f'models/{model_name}'), exist_ok=True)
    pickle.dump(btm, open( f'models/{model_name}/model', 'wb'))
    
    #saving cluster
    y = np.argmax(topics, axis=1)
    groupings = {}
    for i in np.unique(y):
        groupings[str(i)] = []
    for idx, d in enumerate(data):
        groupings[str(y[idx])].append(d)
    with open(join(base_path, f'models/{model_name}/clusters.json'), 'w') as f:
        json.dump(groupings, f, indent=4)

btm = pickle.load(open( f'models/{model_name}/model', 'rb'))

def predict(text):
    # vectorize inputs
    X = vec.fit_transform(text).toarray()
    biterms = vec_to_biterms(text)
    
    #predict topic
    topics = btm.transform(biterms)
    return np.argmax(topics, axis=1)


