import pandas as pd
import numpy as np
import sklearn
import sys
import json
import os
import glob
from os.path import join, dirname, isfile

# danger
sys.path.append('.')
from extractors.raw_count import raw_count_topics
from extractors.intracluster_proportion import intracluster_proportion_topics
from extractors.centroid_spread import centroid_spread_topics
from extractors.word2vec import word2vec_topics
from extractors.centroid_zscore import centroid_zscore_topics

topic_extractors = {
    'raw_count': raw_count_topics,
    'intracluster_proportion': intracluster_proportion_topics,
    'centroid_spread': centroid_spread_topics,
    'word2vec': word2vec_topics,
    'centroid_zscore': centroid_zscore_topics,
}
topic_extractor = None

base_path = dirname(__file__)
model_name = os.path.basename(__file__).split('.')[0]

if len(sys.argv) < 3:
    print('python3 regression.py [model] [topic extractor]')
    print('Model options:')
    for path in glob.glob(join(base_path, '../models/*.py')):
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

with open(join(base_path, f'../models/models/{model_name}/clusters.json'), 'r') as f:
    clusters = json.load(f)
topics, cluster_topics = topic_extractor(clusters)

# groupings = {}
# for label in np.unique(y):
#     groupings[str(label)] = {
#         'keywords': list(model.topic_keywords[label]),
#         'weighted_keywords': list(model.weighted_topic_keywords[label]),
#         'articles': [],
#     }
#     print('Topic keywords:', list(model.topic_keywords[label]))
#     print('Topic weighted keywords:', list(model.topic_keywords[label]))
#     print('')

# # add keywords...
# for idx, sentence in enumerate(data_sentences):
#     label = y[idx]
#     groupings[str(label)]['articles'].append(sentence)

with open(join(base_path, f'groupings/{model_name}.json'), 'w') as f:
    json.dump(cluster_topics, f, indent=4)
