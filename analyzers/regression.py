import numpy as np
import sklearn
import sys
from sklearn.feature_extraction.text import CountVectorizer
import json
import os
import glob
from os.path import join, dirname, isfile
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# danger
sys.path.append('.')
from extractors.raw_count import raw_count_topics
from extractors.intracluster_proportion import intracluster_proportion_topics
from extractors.centroid_spread import centroid_spread_topics

topic_extractors = {
    'raw_count': raw_count_topics,
    'intracluster_proportion': intracluster_proportion_topics,
    'centroid_spread': centroid_spread_topics,
}
topic_extractor = None

base_path = dirname(__file__)
model_name = os.path.basename(__file__).split('.')[0]

if len(sys.argv) < 3 or sys.argv[2] not in topic_extractors:
    print('python3 regression.py [model] [topic extractor]')
    print('Model options:')
    for path in glob.glob(join(base_path, '../models/*.py')):
        print(os.path.basename(path).split('.')[0])
    print('\nTopic extractors:')
    for path in glob.glob(join(base_path, '../extractors/*.py')):
        print(os.path.basename(path).split('.')[0])
    # list models
    sys.exit()

topic_extractor = topic_extractors[sys.argv[2]]

# models are dependent on featurizers
model_name = sys.argv[1]
print(f'Analyzing {model_name} on {sys.argv[2]}')

# import stuff from correct file
# specifically featurizer_name, predict, and topics
import importlib.util
model_path = join(base_path, f'../models/{model_name}.py')
spec = importlib.util.spec_from_file_location("model", model_path)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

# get featurized data + keyword only featurized data
data_sentences = [' '.join(text).lower() for text in model.data]

# get true labels
y = model.predict(data_sentences)

# get topics
def intersect(input_text, topics):
    return [i for i in input_text if i in topics]
with open(join(base_path, f'../models/models/{model_name}/clusters.json'), 'r') as f:
    clusters = json.load(f)
topics, cluster_topics = topic_extractor(clusters)
stripped_data = [' '.join(intersect(text.split(' '), topics)) for idx, text in enumerate(data_sentences)]

# get featurized data
vectorizer = CountVectorizer()
X_base = vectorizer.fit_transform(data_sentences)
X_keywords = vectorizer.fit_transform(stripped_data)

iters = 5
# set up classifiers
lr = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)

# predict
base_scores = np.array([])
keywords_scores = np.array([])
print('Beginning cross validation on logistic regression...')
for i in range(iters):
    X_base, X_keywords, y = sklearn.utils.shuffle(X_base, X_keywords, y)
    base_scores = np.concatenate([base_scores, cross_val_score(lr, X_base, y, cv=5, scoring='accuracy')])
    keywords_scores = np.concatenate([keywords_scores, cross_val_score(lr, X_keywords, y, cv=5, scoring='accuracy')])
    print(f'    Completed iter {i + 1}/{iters}')

print('featurized data', np.mean(base_scores))
print('keywords only', np.mean(keywords_scores))
print('overall score - keywords score', np.mean(base_scores) - np.mean(keywords_scores))
print('standard deviation difference', np.sqrt(np.std(base_scores)**2 + np.std(keywords_scores)**2))
