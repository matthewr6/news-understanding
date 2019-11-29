# @ sasankh

import collections
import numpy as np

from sklearn.linear_model import LogisticRegression

import pandas as pd

top_n_words = 25

# trains a logistic regression classifier and extracts the keywords based on the 25 highest magnitude weights

def log_reg_topics(groupings):
    topics = {}
    for label, sentences in groupings.items():
        if label not in topics:
            topics[label] = collections.defaultdict(int)
        for words in sentences:
            for word in words:
                if len(word) > 0:
                    topics[label][word] += 1


    all_key_words = set()
    cluster_words = {}

    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial')

    df = pd.DataFrame.from_dict(topics, orient='index', dtype=int).fillna(0.0)
    logreg.fit(df, list(topics.keys()))
    print('Trained Classifier')

    weights = abs(logreg.coef_)
    for index, label, in enumerate(topics.keys()):
        ind = np.argpartition(weights[index], -top_n_words)[-top_n_words:]
        kw = []
        for i in ind:
            kw.append(df.columns[i])
        all_key_words.update(kw)
        cluster_words[label] = kw

    return (all_key_words, cluster_words)
