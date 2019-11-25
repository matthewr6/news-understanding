import collections
import numpy as np
import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

top_n_words = 25

# interesting, but not quite working...

def word2vec_topics(groupings):
    all_text = []
    num_clusters = len(groupings.keys())
    centers = [[]] * num_clusters
    for label in groupings:
        all_text += groupings[label]

    word2vec = Word2Vec(all_text, size=250, min_count=1)

    for label in groupings:
        tfidf = TfidfVectorizer()
        tfidf.fit([' '.join(s) for s in groupings[label]])
        transformed_sentences = tfidf.transform([' '.join(s) for s in groupings[label]]).toarray()
        for idx, sentence in enumerate(groupings[label]):
            word_weights = transformed_sentences[idx]
            vec = [word2vec.wv[word] * word_weights[tfidf.vocabulary_[word]] for word in sentence if word in word2vec.wv and word in tfidf.vocabulary_]
            if len(vec) == 0:
                vec = np.zeros((1, 250))
            centers[int(label)].append(np.mean(vec, axis=0))
        centers[int(label)] = np.mean(centers[int(label)], axis=0)

    cluster_words = {}
    all_words = []
    for label in groupings:
        cluster_words[str(label)] = [w[0] for w in word2vec.wv.similar_by_vector(centers[int(label)], topn=top_n_words)]
        all_words += cluster_words[str(label)]

    return (set(all_words), cluster_words)
