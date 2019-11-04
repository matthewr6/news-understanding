# Set of make commands that wrap around the python
# scripts.

# Download news data from newsapi.org
dataset:
	python3 datasets/pull.py
	python3 datasets/compile.py
	python3 datasets/flatten.py

# add custom args or whatever for these
featurize/%:
	python3 featurizers/$*.py

model/%:
	python3 models/$*.py

analyze/%:
	python3 analyzers/regression.py $*
