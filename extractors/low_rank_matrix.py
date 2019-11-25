import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

top_n_words = 25

# Rough idea.

def decrease_rank(M, new_rank):
    U, S, V = np.linalg.svd(M)
    low_rank_mat = np.zeros((U.shape[0], V.shape[0]))
    for i in range(new_rank):
        low_rank_mat += s[i] * np.outer(u.T[i], v[i])
    return low_rank_mat

def low_rank_matrix_topics(groupings):
    vectorizer = CountVectorizer()
    all_text = []
    num_clusters = len(groupings.keys())
    centers = [0] * num_clusters
    for label in groupings:
        all_text += [' '.join(line) for line in groupings[label]]
    vectorizer.fit(all_text)
    for label in groupings:
        centers[int(label)] = vectorizer.transform([' '.join(line) for line in groupings[label]]).toarray()
    for idx in range(len(centers)):
        centers[idx] = np.stack(centers[idx], axis=0)
        centers[idx] = decrease_rank(centers[idx], top_n_words)