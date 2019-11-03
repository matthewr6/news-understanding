# Set of make commands that wrap around the python
# scripts.

# Download news data from newsapi.org
dataset:
	python3 datasets/pull.py
	python3 datasets/compile.py
	python3 datasets/flatten.py

featurize:
	python3 featurizers/featurize.py

model:
	python3 models/model.py

analyze:
	python3 analyzers/analyze.py
