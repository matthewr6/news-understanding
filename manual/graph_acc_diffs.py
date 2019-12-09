# featurizers are x axis
# colors are extractors
# order is raw count, intra, centroid, log reg

import json
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from os.path import join, dirname, isfile
base_path = dirname(__file__)

fname = 'accuracy_diffs'
with open(join(base_path, f'{fname}.json'), 'rb') as f:
    data = json.load(f)

new_data = {}
order = ['Count', 'Pseudo TF-IDF', 'Centroid', 'Logistic Regression']
ks = ['simple', 'no_stopwords', 'spacy_nouns', 'median_word_length']
featurizers = ['Simple', 'No Stopwords', 'Nouns', 'Word Length']
for idx, label in enumerate(order):
    a = []
    for k in ks:
        a.append(data[k][idx])
    new_data[label] = 1 - np.array(a)

print(new_data)

barWidth = 0.12
r1 = np.arange(len(new_data['Count']))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

rs = [r1, r2, r3, r4]

colors = ['#9ED5CD', '#44A7CB', '#2E62A1', '#192574']

for idx, o in enumerate(order):
    plt.bar(rs[idx], new_data[o], color=colors[idx], width=barWidth, edgecolor='white', label=o)

plt.xlabel('Featurizer', fontweight='bold')
plt.xticks([r + (barWidth * 1.5) for r in range(len(new_data['Count']))], featurizers)

plt.ylim(0.95, 1)

# plt.legend(bbox_to_anchor=(1.04,0.65), loc="upper left", borderaxespad=0)
plt.legend(ncol=2)

plt.savefig(join(base_path, f'{fname}.png'), bbox_inches="tight")
