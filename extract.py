from datetime import datetime, timedelta
import numpy as np
import json
import requests
import os
import regex as re

import spacy
nlp = spacy.load("en_core_web_sm")


from os.path import join, dirname, isfile
path = dirname(__file__)

with open(join(path, 'raw_data/news.json'), 'r') as f:
    data = json.load(f)


def should_include_word(word):
    return len(word) >= min_len

min_len = 4

def parse_article(article):
    text = ''
    if 'title' in article and article['title']:
            text = article['title']
    if 'description' in article and article['description']:
            text += ' ' + article['description']
    doc = nlp(text.strip())
    # print([t for t in doc])
    tokens = [token.lemma_ for token in doc if 'NN' in token.tag_]
    return tokens
    # text = re.sub("\p{P}+", "", text).lower()
    # text = text.split(' ')
    # text = filter(should_include_word, text)
    # return list(text)


parsed_data = {}
for date in data:
    articles = data[date]['articles']
    parsed_data[date] = [parse_article(a) for a in articles]


with open(join(path, 'parsed_data/news_text.json'), 'w') as f:
    json.dump(parsed_data, f, ensure_ascii=False)
