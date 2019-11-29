# @ sasankh

import collections
import numpy as np

from sklearn.linear_model import LogisticRegression

from itertools import combinations

import pandas as pd

top_n_words = 25


def log_reg_topics(groupings):
    topics = {}
    for label, sentences in groupings.items():
        if label not in topics:
            topics[label] = collections.defaultdict(int)
        for words in sentences:
            for word in words:
                if len(word) > 0:
                    topics[label][word] += 1


    all_key_words = []
    cluster_words = {}

    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial')

    df = pd.DataFrame.from_dict(topics, orient='index', dtype=int).fillna(0.0)
    logreg.fit(df, list(topics.keys()))
    print('Trained Classifier')


    for label, words in topics.items():




        comb = list(combinations(unique_words, top_n_words))

        highest_prob = 0
        highest_comb = []
        for w in comb:
            res = clf.predict(w)
            if res == label and clf.predict_proba(w) > highest_prob:
                highest_prob = clf.predict_proba(w)
                highest_comb = w

        all_key_words.append(highest_comb)
        cluster_words[label] = highest_comb

    return (set(all_key_words), cluster_words)