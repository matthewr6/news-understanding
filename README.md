# Instructions

Aside from the NewsAPI API key, all relevant data is included in this repository.

## Data collection

The dataset is frozen.  Running `make dataset` with a live dataset would update the current dataset in `datasets/data.json`; however, the dataset was frozen to allow for consistent experiments.

## Preprocessing

All preprocessors are stored in `featurizers/` and preprocessed data in `featurizers/featurized_data/`.  To run a given featurizer, run `make featurize/<featurizer name>`, or `make featurize/all` to run all preprocessors.

## Models

Due to the way our infrastructure was designed, models were not independent of preprocessors.  So, a given model file in `models/` combines a clustering model, its parameters, and which preprocessed data it ingests.  Running any `models/<model name>.py` trains and saves the model into `models/models/`.  Note that the `biterm` and `tkm` models may not be runnable because they require specific installation and setup from https://github.com/markoarnauto/biterm and https://github.com/JohnTailor/tkm, respectively (see the Miscellaneous section for more info).

If using the Makefile, run `make model/<model name>` to train a given model or `make model/all` to train all 10-cluster K-means models.

## Analysis

First, for quantitative analysis of the clustering models, run `python analyzers/cluster_score.py <model name>` (or, to run it on across K-means models using different cluster numbers, run `make clusterscore/clusternums`).  Then, for qualitative analysis, run `python analyzers/visualize.py <model name>` (or, again, `make visualize/all` for 10-cluster k-means visualization).  Output scatterplots use the first three PCA dimensions and are stored in `analyzers/graphs/`.

Then, to analyze the results of the keyword extractors, cluster keywords can be examined by running `python analyzers/group.py <model name> <extractor name>` (or for 10-cluster k-means: `make group/all extractor=<extractor name>`).  The cluster keywords are stored in `groupings/`.  The effectiveness of the cluster keywords can be quantitatively evaluated using `python analyzers/regression.py <model name> <extractor name>` (or, again, `make analyze/exp10` for 10-cluster k-means).

## Miscellaneous

After running `make analyze/exp10`, the difference values were manually copied into `manual/accuracy_diffs.json` and `manual/graph_acc_diffs.py` was run to produce the `manual/accuracy_differences.png`.

Both repositories mentioned earlier were `git clone`d into `vendor/` and `pip install .` was ran from within the repository home directory.  A Cython environment was used for installing and training both biterm and tkm models.