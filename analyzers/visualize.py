import pandas as pd
import numpy as np
import sklearn
import sys
# import nltk
import gensim.corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.preprocessing import normalize
# import pickle
import json
import os
import glob
from os.path import join, dirname, isfile
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

import seaborn as sns

base_path = dirname(__file__)
model_name = os.path.basename(__file__).split('.')[0]

if len(sys.argv) < 2:
    print('python3 visualize.py [model]')
    print('Model options:')
    for path in glob.glob(join(base_path, '../models/*.py')):
        print(os.path.basename(path).split('.')[0])
    # list models
    sys.exit()

# models are dependent on featurizers
model_name = sys.argv[1]

# import stuff from correct file
# specifically featurizer_name, predict
import importlib.util
model_path = join(base_path, f'../models/{model_name}.py')
spec = importlib.util.spec_from_file_location("model", model_path)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

# get featurized data + keyword only featurized data
data_sentences = [' '.join(text).lower() for text in model.data]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_sentences)
y = model.predict(data_sentences)

reduced_data = PCA(n_components=3).fit_transform(X.todense())

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=y, s=1)
plt.savefig(f"{base_path}/graphs/{sys.argv[1]}_3d.png")

plt.clf()
fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(reduced_data[:, 0], reduced_data[:, 1], c=y, s=1)
axs[0, 0].set_title('Dims 1 x 2')
axs[0, 1].scatter(reduced_data[:, 0], reduced_data[:, 2], c=y, s=1)
axs[0, 1].set_title('Dims 1 x 3')
axs[1, 0].scatter(reduced_data[:, 1], reduced_data[:, 2], c=y, s=1)
axs[1, 0].set_title('Dims 2 x 3')
plt.savefig(f"{base_path}/graphs/{sys.argv[1]}.png")
