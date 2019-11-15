# Set of make commands that wrap around the python
# scripts.

help:
	cat Makefile

list:
	@echo "Featurizers:"
	@ls featurizers/ | grep -v "pycache" | grep "py"
	@echo "\nModels:"
	@ls models/ | grep -v "pycache" | grep "py"

# Download news data from newsapi.org
dataset:
	python3 datasets/pull.py
	python3 datasets/compile.py
	python3 datasets/flatten.py

featurize/all:
	time python3 featurizers/simple.py
	time python3 featurizers/no_stopwords.py
	time python3 featurizers/median_word_length.py
	time python3 featurizers/spacy_nouns.py

# add custom args or whatever for these
featurize/%:
	time python3 featurizers/$*.py

model/all:
	time python3 models/kmeans_10topics_simple.py
	time python3 models/kmeans_10topics_no_stopwords.py
	time python3 models/kmeans_10topics_spacy_nouns.py
	time python3 models/kmeans_10topics_median_word_length.py

model/%:
	time python3 models/$*.py $(featurizer)

analyze/all:
	python3 analyzers/regression.py kmeans_10topics_simple
	python3 analyzers/regression.py kmeans_10topics_no_stopwords
	python3 analyzers/regression.py kmeans_10topics_spacy_nouns
	python3 analyzers/regression.py kmeans_10topics_median_word_length
# 	python3 analyzers/regression.py kmeans_10topics_ngram_2

analyze/%:
	python3 analyzers/regression.py $*

visualize/all:
	python3 analyzers/visualize.py kmeans_10topics_simple
	python3 analyzers/visualize.py kmeans_10topics_no_stopwords
	python3 analyzers/visualize.py kmeans_10topics_spacy_nouns
	python3 analyzers/visualize.py kmeans_10topics_median_word_length

visualize/%:
	python3 analyzers/visualize.py $*

group/all:
	python3 analyzers/group.py kmeans_10topics_simple
	python3 analyzers/group.py kmeans_10topics_no_stopwords
	python3 analyzers/group.py kmeans_10topics_spacy_nouns
	python3 analyzers/group.py kmeans_10topics_median_word_length
	