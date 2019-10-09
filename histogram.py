import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import numpy as np
import json
import requests
import os

from os.path import join, dirname, isfile
path = dirname(__file__)

with open(join(path, 'news_flat.json'), 'r') as f:
    data = json.load(f)

words = [item for sublist in data for item in sublist]

from collections import Counter


print(len(words))

counts = dict(Counter(words).most_common(100))

labels, values = zip(*counts.items())

# sort your values in descending order
indSort = np.argsort(values)[::-1]

# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]

indexes = np.arange(len(labels))

bar_width = 0.35

plt.bar(indexes, values)

# add labels
plt.xticks(indexes + bar_width, labels)
print(labels)
plt.savefig('word_histogram.png')