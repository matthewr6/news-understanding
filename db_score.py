from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score

def db_score(groupings):
    flat_list = []
    labels = []
    for i, sublist in enumerate(list(groupings.values())):
        for item in sublist:
            flat_list.append(item)
            labels.append(i)
    joined_list = []
    for a in flat_list:
        joined_list.append(" ".join(a))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(joined_list).toarray()
    return(davies_bouldin_score(X,labels))