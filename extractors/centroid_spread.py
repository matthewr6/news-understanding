import collections
import numpy as np
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer

top_n_words = 25

# uses top 250 dimensions where centroids have
# highest standard deviation and assigns
# those dimensions to the most relevent
# clusters based on cosine dist

def centroid_spread_topics(groupings):
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
    deviations = np.std(centers, axis=0)
    highest_deviations = np.argsort(deviations)[::-1]
    possible_word_vecs = np.identity(deviations.shape[0])
    word_vecs = [possible_word_vecs[word_id] for word_id in highest_deviations]
    words = [w[0] for w in vectorizer.inverse_transform(word_vecs)]
    topic_words = collections.defaultdict(set)
    number_to_assign = num_clusters * top_n_words
    all_words = set()
    for i in range(number_to_assign):
        relevances = [center[highest_deviations[i]] for center in centers]
        assignment = np.argmax(relevances)
        topic_words[str(assignment)].add(words[i])
        all_words.add(words[i])
    topic_words = dict(topic_words)
    for label in topic_words:
        topic_words[label] = list(topic_words[label])
    return (all_words, topic_words)