import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

top_n_words = 25

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=top_n_words):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results

def tfidf_topics(groupings):
    topics = []
    labels = []
    for label, sentences in groupings.items():
        idf = []
        for sentence in sentences:
            idf+=sentence
        topics.append(idf)
        labels.append(label)
    topics_input = []
    for sentences in topics:
        topics_input.append(" ".join(sentences))
    cv=CountVectorizer(max_df=0.9,max_features=10000)
    word_count_vector=cv.fit_transform(topics_input)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    feature_names=cv.get_feature_names()
    all_key_words = []
    cluster_words = {}
    for i in range(len(topics_input)):
        doc=topics_input[i]
        tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
        sorted_items=sort_coo(tf_idf_vector.tocoo())
        keywords=extract_topn_from_vector(feature_names,sorted_items)
        cluster_words[labels[i]] = list(keywords.keys())
        all_key_words += list(keywords.keys())
    return (set(all_key_words), cluster_words)
