import collections
import numpy as np
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer

top_n_words = 25

# bad because only takes into account how
# unique word is - not whether it's
# relevant

def centroid_zscore_topics(groupings):
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
        centers[idx] = np.mean(centers[idx], axis=0)
    centers = np.array(centers)
    average_center = np.repeat(np.mean(centers, axis=0), centers.shape[0], axis=0).reshape((centers.shape[0], centers.shape[1]))
    deviations = np.std(centers, axis=0)
    z_scores = np.divide(centers - average_center, deviations)
    z_scores = np.multiply(z_scores, centers)
    max_zscores = np.argsort(z_scores, axis=1)
    possible_word_vecs = np.identity(deviations.shape[0])
    words = [w[0] for w in vectorizer.inverse_transform(possible_word_vecs)]
    topic_words = collections.defaultdict(set)
    all_words = set()
    for i in range(max_zscores.shape[0]):
        topic_words[str(i)] = [words[w] for w in max_zscores[i][:top_n_words]]
    topic_words = dict(topic_words)
    for label in topic_words:
        topic_words[label] = list(topic_words[label])
    return (all_words, topic_words)