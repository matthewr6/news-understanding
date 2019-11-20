import collections

top_n_words = 25

def raw_count_topics(groupings):
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
    for label, words in topics.items():
        if label not in cluster_words:
            cluster_words[label] = set()
        c = collections.Counter(words)
        for w in c.most_common(top_n_words):
            all_key_words.append(w[0])
            cluster_words[label].add(w[0])
    for label in cluster_words:
        cluster_words[label] = list(cluster_words[label])
    return (set(all_key_words), cluster_words)
