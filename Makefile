# Set of make commands that wrap around the python
# scripts.

help:
	cat Makefile

list:
	@echo "Featurizers:"
	@ls featurizers/ | grep -v "pycache" | grep "py"
	@echo "\nModels:"
	@ls models/ | grep -v "pycache" | grep "py"
	@echo "\nExtractors:"
	@ls extractors/ | grep -v "pycache" | grep "py"

# Download news data from newsapi.org
dataset:
	python3 datasets/pull.py
	python3 datasets/compile.py
	python3 datasets/flatten.py

###############
# FEATURIZERS #
###############

featurize/all:
	time python3 featurizers/simple.py
	time python3 featurizers/no_stopwords.py
	time python3 featurizers/median_word_length.py
	time python3 featurizers/spacy_nouns.py

featurize/%:
	time python3 featurizers/$*.py

##########
# MODELS #
##########

model/all:
	time python3 models/kmeans_10topics_simple.py
	time python3 models/kmeans_10topics_no_stopwords.py
	time python3 models/kmeans_10topics_spacy_nouns.py
	time python3 models/kmeans_10topics_median_word_length.py

model/%:
	time python3 models/$*.py $(featurizer)

############
# ANALYSIS #
############

analyze/all:
	python3 analyzers/regression.py kmeans_10topics_simple $(extractor)
	python3 analyzers/regression.py kmeans_10topics_no_stopwords $(extractor)
	python3 analyzers/regression.py kmeans_10topics_spacy_nouns $(extractor)
	python3 analyzers/regression.py kmeans_10topics_median_word_length $(extractor)

analyze/%:
	python3 analyzers/regression.py $* $(extractor)

visualize/all:
	python3 analyzers/visualize.py kmeans_10topics_simple
	python3 analyzers/visualize.py kmeans_10topics_no_stopwords
	python3 analyzers/visualize.py kmeans_10topics_spacy_nouns
	python3 analyzers/visualize.py kmeans_10topics_median_word_length

visualize/%:
	python3 analyzers/visualize.py $*

group/all:
	python3 analyzers/group.py kmeans_10topics_simple $(extractor)
	python3 analyzers/group.py kmeans_10topics_no_stopwords $(extractor)
	python3 analyzers/group.py kmeans_10topics_spacy_nouns $(extractor)
	python3 analyzers/group.py kmeans_10topics_median_word_length $(extractor)

group/%:
	python3 analyzers/group.py $* $(extractor)
	