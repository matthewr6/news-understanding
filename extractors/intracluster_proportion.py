import collections

top_n_words = 25
possible_top_words = 100

# proportion in cluster^2 / total proportion

def normalize_dict(d):
    total = 0
    for k, v in d.items():
        total += v
    for k in d:
        d[k] /= total
    return d

def intracluster_proportion_topics(groupings):
    topics = {}
    for label, sentences in groupings.items():
        if label not in topics:
            topics[label] = collections.defaultdict(int)
        for words in sentences:
            for word in words:
                if len(word) > 0:
                    topics[label][word] += 1
    normalized_terms = {}
    totals = collections.defaultdict(float)
    for label, words in topics.items():
        c = collections.Counter(words)
        normalized_terms[label] = normalize_dict(dict(c.most_common(possible_top_words)))
        for term, proportion in normalized_terms[label].items():
            totals[term] += proportion
    for label, terms in normalized_terms.items():
        for term, value in terms.items():
            normalized_terms[label][term] *= value / totals[term]
    all_keywords = []
    cluster_words = {}
    for label, words in normalized_terms.items():
        if label not in cluster_words:
            cluster_words[label] = []
        c = collections.Counter(words)
        for w in c.most_common(top_n_words):
            all_keywords.append(w[0])
            if w[0] not in cluster_words[label]:
                cluster_words[label].append(w[0])
    # for label in cluster_words:
    #     cluster_words[label] = list(cluster_words[label])
    return (set(all_keywords), cluster_words)
