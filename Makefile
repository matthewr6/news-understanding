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
	@echo "Dataset is frozen"
# 	python3 datasets/pull.py
# 	python3 datasets/compile.py
# 	python3 datasets/flatten.py

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

model/all8:
	time python3 models/kmeans_8topics_simple.py
	time python3 models/kmeans_8topics_no_stopwords.py
	time python3 models/kmeans_8topics_spacy_nouns.py
	time python3 models/kmeans_8topics_median_word_length.py

model/all12:
	time python3 models/kmeans_12topics_simple.py
	time python3 models/kmeans_12topics_no_stopwords.py
	time python3 models/kmeans_12topics_spacy_nouns.py
	time python3 models/kmeans_12topics_median_word_length.py

model/tkm:
# 	time python3 models/tkm_10topics_simple.py
	time python3 models/tkm_10topics_no_stopwords.py
	time python3 models/tkm_10topics_spacy_nouns.py
	time python3 models/tkm_15topics_no_stopwords.py
	time python3 models/tkm_20topics_spacy_nouns.py
# 	time python3 models/tkm_10topics_median_word_length.py

model/lda:
	time python3 models/lda_10topics_simple.py
	time python3 models/lda_10topics_no_stopwords.py
	time python3 models/lda_10topics_median_word_length.py
	time python3 models/lda_10topics_spacy_nouns.py
# 	time python3 models/lda_15topics_simple.py
# 	time python3 models/lda_20topics_simple.py

model/%:
	time python3 models/$*.py

############
# ANALYSIS #
############

analyze/all:
	python3 analyzers/regression.py kmeans_10topics_simple $(extractor)
	python3 analyzers/regression.py kmeans_10topics_no_stopwords $(extractor)
	python3 analyzers/regression.py kmeans_10topics_spacy_nouns $(extractor)
	python3 analyzers/regression.py kmeans_10topics_median_word_length $(extractor)

analyze/exp8:
	python3 analyzers/regression.py kmeans_8topics_simple raw_count
	python3 analyzers/regression.py kmeans_8topics_no_stopwords raw_count
	python3 analyzers/regression.py kmeans_8topics_spacy_nouns raw_count
	python3 analyzers/regression.py kmeans_8topics_median_word_length raw_count
	python3 analyzers/regression.py kmeans_8topics_simple log_reg
	python3 analyzers/regression.py kmeans_8topics_no_stopwords log_reg
	python3 analyzers/regression.py kmeans_8topics_spacy_nouns log_reg
	python3 analyzers/regression.py kmeans_8topics_median_word_length log_reg
	python3 analyzers/regression.py kmeans_8topics_simple centroid_spread
	python3 analyzers/regression.py kmeans_8topics_no_stopwords centroid_spread
	python3 analyzers/regression.py kmeans_8topics_spacy_nouns centroid_spread
	python3 analyzers/regression.py kmeans_8topics_median_word_length centroid_spread
	python3 analyzers/regression.py kmeans_8topics_simple intracluster_proportion
	python3 analyzers/regression.py kmeans_8topics_no_stopwords intracluster_proportion
	python3 analyzers/regression.py kmeans_8topics_spacy_nouns intracluster_proportion
	python3 analyzers/regression.py kmeans_8topics_median_word_length intracluster_proportion

analyze/logreg8:
	python3 analyzers/regression.py kmeans_8topics_simple log_reg
	python3 analyzers/regression.py kmeans_8topics_no_stopwords log_reg
	python3 analyzers/regression.py kmeans_8topics_spacy_nouns log_reg
	python3 analyzers/regression.py kmeans_8topics_median_word_length log_reg

analyze/logreg10:
	python3 analyzers/regression.py kmeans_10topics_simple log_reg
	python3 analyzers/regression.py kmeans_10topics_no_stopwords log_reg
	python3 analyzers/regression.py kmeans_10topics_spacy_nouns log_reg
	python3 analyzers/regression.py kmeans_10topics_median_word_length log_reg


analyze/exp10:
	python3 analyzers/regression.py kmeans_10topics_simple raw_count
	python3 analyzers/regression.py kmeans_10topics_no_stopwords raw_count
	python3 analyzers/regression.py kmeans_10topics_spacy_nouns raw_count
	python3 analyzers/regression.py kmeans_10topics_median_word_length raw_count
	python3 analyzers/regression.py kmeans_10topics_simple log_reg
	python3 analyzers/regression.py kmeans_10topics_no_stopwords log_reg
	python3 analyzers/regression.py kmeans_10topics_spacy_nouns log_reg
	python3 analyzers/regression.py kmeans_10topics_median_word_length log_reg
	python3 analyzers/regression.py kmeans_10topics_simple centroid_spread
	python3 analyzers/regression.py kmeans_10topics_no_stopwords centroid_spread
	python3 analyzers/regression.py kmeans_10topics_spacy_nouns centroid_spread
	python3 analyzers/regression.py kmeans_10topics_median_word_length centroid_spread
	python3 analyzers/regression.py kmeans_10topics_simple intracluster_proportion
	python3 analyzers/regression.py kmeans_10topics_no_stopwords intracluster_proportion
	python3 analyzers/regression.py kmeans_10topics_spacy_nouns intracluster_proportion
	python3 analyzers/regression.py kmeans_10topics_median_word_length intracluster_proportion

clusterscore/kmeans10:
	python3 analyzers/cluster_score.py kmeans_10topics_simple
	python3 analyzers/cluster_score.py kmeans_10topics_no_stopwords
	python3 analyzers/cluster_score.py kmeans_10topics_spacy_nouns
	python3 analyzers/cluster_score.py kmeans_10topics_median_word_length

clusterscore/exp10:
	python3 analyzers/cluster_score.py kmeans_10topics_simple raw_count
	python3 analyzers/cluster_score.py kmeans_10topics_no_stopwords raw_count
	python3 analyzers/cluster_score.py kmeans_10topics_spacy_nouns raw_count
	python3 analyzers/cluster_score.py kmeans_10topics_median_word_length raw_count
	python3 analyzers/cluster_score.py kmeans_10topics_simple centroid_spread
	python3 analyzers/cluster_score.py kmeans_10topics_no_stopwords centroid_spread
	python3 analyzers/cluster_score.py kmeans_10topics_spacy_nouns centroid_spread
	python3 analyzers/cluster_score.py kmeans_10topics_median_word_length centroid_spread
	python3 analyzers/cluster_score.py kmeans_10topics_simple intracluster_proportion
	python3 analyzers/cluster_score.py kmeans_10topics_no_stopwords intracluster_proportion
	python3 analyzers/cluster_score.py kmeans_10topics_spacy_nouns intracluster_proportion
	python3 analyzers/cluster_score.py kmeans_10topics_median_word_length intracluster_proportion

analyze/exp12:
	python3 analyzers/regression.py kmeans_12topics_simple raw_count
	python3 analyzers/regression.py kmeans_12topics_no_stopwords raw_count
	python3 analyzers/regression.py kmeans_12topics_spacy_nouns raw_count
	python3 analyzers/regression.py kmeans_12topics_median_word_length raw_count
	python3 analyzers/regression.py kmeans_12topics_simple centroid_spread
	python3 analyzers/regression.py kmeans_12topics_no_stopwords centroid_spread
	python3 analyzers/regression.py kmeans_12topics_spacy_nouns centroid_spread
	python3 analyzers/regression.py kmeans_12topics_median_word_length centroid_spread
	python3 analyzers/regression.py kmeans_12topics_simple intracluster_proportion
	python3 analyzers/regression.py kmeans_12topics_no_stopwords intracluster_proportion
	python3 analyzers/regression.py kmeans_12topics_spacy_nouns intracluster_proportion
	python3 analyzers/regression.py kmeans_12topics_median_word_length intracluster_proportion

clusterscore/exp12:
	python3 analyzers/cluster_score.py kmeans_12topics_simple raw_count
	python3 analyzers/cluster_score.py kmeans_12topics_no_stopwords raw_count
	python3 analyzers/cluster_score.py kmeans_12topics_spacy_nouns raw_count
	python3 analyzers/cluster_score.py kmeans_12topics_median_word_length raw_count
	python3 analyzers/cluster_score.py kmeans_12topics_simple centroid_spread
	python3 analyzers/cluster_score.py kmeans_12topics_no_stopwords centroid_spread
	python3 analyzers/cluster_score.py kmeans_12topics_spacy_nouns centroid_spread
	python3 analyzers/cluster_score.py kmeans_12topics_median_word_length centroid_spread
	python3 analyzers/cluster_score.py kmeans_12topics_simple intracluster_proportion
	python3 analyzers/cluster_score.py kmeans_12topics_no_stopwords intracluster_proportion
	python3 analyzers/cluster_score.py kmeans_12topics_spacy_nouns intracluster_proportion
	python3 analyzers/cluster_score.py kmeans_12topics_median_word_length intracluster_proportion


clusterscore/clusternums:
	python3 analyzers/cluster_score.py kmeans_8topics_simple
	python3 analyzers/cluster_score.py kmeans_8topics_no_stopwords
	python3 analyzers/cluster_score.py kmeans_8topics_spacy_nouns
	python3 analyzers/cluster_score.py kmeans_8topics_median_word_length
	@echo ""
	python3 analyzers/cluster_score.py kmeans_10topics_simple
	python3 analyzers/cluster_score.py kmeans_10topics_no_stopwords
	python3 analyzers/cluster_score.py kmeans_10topics_spacy_nouns
	python3 analyzers/cluster_score.py kmeans_10topics_median_word_length
	@echo ""
	python3 analyzers/cluster_score.py kmeans_12topics_simple
	python3 analyzers/cluster_score.py kmeans_12topics_no_stopwords
	python3 analyzers/cluster_score.py kmeans_12topics_spacy_nouns
	python3 analyzers/cluster_score.py kmeans_12topics_median_word_length



# lda 10 topics best
analyze/lda:
	python3 analyzers/regression.py lda_10topics_simple $(extractor)
	python3 analyzers/regression.py lda_10topics_no_stopwords $(extractor)
	python3 analyzers/regression.py lda_10topics_spacy_nouns $(extractor)
	python3 analyzers/regression.py lda_10topics_median_word_length $(extractor)

analyze/%:
	python3 analyzers/regression.py $* $(extractor)

visualize/kmeans:
	python3 analyzers/visualize.py kmeans_5topics_median_word_length
	python3 analyzers/visualize.py kmeans_8topics_median_word_length
	python3 analyzers/visualize.py kmeans_10topics_median_word_length
	python3 analyzers/visualize.py kmeans_12topics_median_word_length
	python3 analyzers/visualize.py kmeans_15topics_median_word_length


visualize/kmeans12:
	python3 analyzers/visualize.py kmeans_12topics_median_word_length
	python3 analyzers/visualize.py kmeans_12topics_simple
	python3 analyzers/visualize.py kmeans_12topics_spacy_nouns
	python3 analyzers/visualize.py kmeans_12topics_no_stopwords



visualize/all:
	python3 analyzers/visualize.py kmeans_10topics_simple
	python3 analyzers/visualize.py kmeans_10topics_no_stopwords
	python3 analyzers/visualize.py kmeans_10topics_spacy_nouns
	python3 analyzers/visualize.py kmeans_10topics_median_word_length

visualize/tkm:
	python3 analyzers/visualize.py tkm_10topics_no_stopwords
	python3 analyzers/visualize.py tkm_10topics_spacy_nouns
	python3 analyzers/visualize.py tkm_15topics_no_stopwords
	python3 analyzers/visualize.py tkm_20topics_spacy_nouns

visualize/lda:
	python3 analyzers/visualize.py lda_10topics_simple 
	python3 analyzers/visualize.py lda_10topics_no_stopwords 
	python3 analyzers/visualize.py lda_10topics_spacy_nouns 
	python3 analyzers/visualize.py lda_10topics_median_word_length 

visualize/%:
	python3 analyzers/visualize.py $*

group/all:
	python3 analyzers/group.py kmeans_10topics_simple $(extractor)
	python3 analyzers/group.py kmeans_10topics_no_stopwords $(extractor)
	python3 analyzers/group.py kmeans_10topics_spacy_nouns $(extractor)
	python3 analyzers/group.py kmeans_10topics_median_word_length $(extractor)

group/all8:
	python3 analyzers/group.py kmeans_8topics_simple raw_count
	python3 analyzers/group.py kmeans_8topics_no_stopwords raw_count
	python3 analyzers/group.py kmeans_8topics_spacy_nouns raw_count
	python3 analyzers/group.py kmeans_8topics_median_word_length raw_count
	python3 analyzers/group.py kmeans_8topics_simple log_reg
	python3 analyzers/group.py kmeans_8topics_no_stopwords log_reg
	python3 analyzers/group.py kmeans_8topics_spacy_nouns log_reg
	python3 analyzers/group.py kmeans_8topics_median_word_length log_reg
	python3 analyzers/group.py kmeans_8topics_simple centroid_spread
	python3 analyzers/group.py kmeans_8topics_no_stopwords centroid_spread
	python3 analyzers/group.py kmeans_8topics_spacy_nouns centroid_spread
	python3 analyzers/group.py kmeans_8topics_median_word_length centroid_spread
	python3 analyzers/group.py kmeans_8topics_simple intracluster_proportion
	python3 analyzers/group.py kmeans_8topics_no_stopwords intracluster_proportion
	python3 analyzers/group.py kmeans_8topics_spacy_nouns intracluster_proportion
	python3 analyzers/group.py kmeans_8topics_median_word_length intracluster_proportion

group/all10:
	python3 analyzers/group.py kmeans_10topics_simple raw_count
	python3 analyzers/group.py kmeans_10topics_no_stopwords raw_count
	python3 analyzers/group.py kmeans_10topics_spacy_nouns raw_count
	python3 analyzers/group.py kmeans_10topics_median_word_length raw_count
	python3 analyzers/group.py kmeans_10topics_simple log_reg
	python3 analyzers/group.py kmeans_10topics_no_stopwords log_reg
	python3 analyzers/group.py kmeans_10topics_spacy_nouns log_reg
	python3 analyzers/group.py kmeans_10topics_median_word_length log_reg
	python3 analyzers/group.py kmeans_10topics_simple centroid_spread
	python3 analyzers/group.py kmeans_10topics_no_stopwords centroid_spread
	python3 analyzers/group.py kmeans_10topics_spacy_nouns centroid_spread
	python3 analyzers/group.py kmeans_10topics_median_word_length centroid_spread
	python3 analyzers/group.py kmeans_10topics_simple intracluster_proportion
	python3 analyzers/group.py kmeans_10topics_no_stopwords intracluster_proportion
	python3 analyzers/group.py kmeans_10topics_spacy_nouns intracluster_proportion
	python3 analyzers/group.py kmeans_10topics_median_word_length intracluster_proportion

group/%:
	python3 analyzers/group.py $* $(extractor)
	