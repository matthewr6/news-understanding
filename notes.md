Featurizers
- Currently have median word length, ngrams, no stopwords, no punctuation, and noun tagging
- Going to add lemmatizing and stemmng
- Currently have 5 (4, ngrams bad) and two potential
- 6 total useful

Models

- K-means - figuring out number of clusters (currently set at 10 arbitrarily)
- Run on 15 and 20 as well?
- Err on the side of more
    - we can discuss quantitatively what clusters should go together based on keywords and then
      use that reduced value as the new number of clusters
- Run LDA experiments on featurizers as well (will be faster than K-means luckily)
- Implement biterm from notebook
- Gaussian Mixture Model?
- Three total - kmeans, lda, biterm (maybe GMM as 4th?)

At this point, 6 x 3 = 18 combinations (or 24)

Topic extraction

- Raw count-based
- Weighted based on relative proportion across topics
- Currently have 2 extractors (very similar outputs), let's try to get to ~5
- pseudo tf-idf but based on clusters rather than documents?

Visualization

- analyzers/visualize.py for graphing clusters (PCA)
- Visualization good for trying to find # clusters, quantitative analysis of clusters
- analyzers/group.py for viewing cluster keywords and articles assigned to cluster
- Good for quantitative analysis of clusters and keywords

Todo

- write README
- Biterm model finalizing + experiments (Matthew)
    - qualitative analysis, explain which one's best
- K-means experiments (different # clusters) (Sasankh)
    - qualitative analysis, explain which one's best
- LDA experiments (different featurizers/clusters) (Matthew)
    - qualitative analysis, explain which one's best
- 2-3 more topic extractors
    - Recall we want to find common terms unique to the cluster
    - Pseudo tf-idf but based on clusters rather than documents (Chris) (tf is within cluster, idf is against outside clusters)
    - Somehow based on centroid vector (largest, average difference between other centroids?) (Matthew)
    - MultinomialNB (Sasankh)
- Brief descriptions/writeups of each featurizer, model (+ extractor), and experiment (if experiment is interesting; from results and visualizations)
- Poster (Sasankh + Chris)
- Start on paper (Chris + Sasankh)

Graphs (matthew?)
- accuracy differences
- for most successful model - plot of clusterings
    - cluster topics
    - examples of articles
