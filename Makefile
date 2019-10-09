# Set of make commands that wrap around the python
# scripts.

# Download news data from newsapi.org
rawdata:
	python3 data.py

# Preprocess data into format that can be fed into model
pipeline:
	python3 extract.py
	python3 flatten.py

# Data inspection/visualization tools
inspect:
	python3 inspect_data.py
	python3 histogram.py
