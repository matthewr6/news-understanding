from datetime import datetime, timedelta
import numpy as np
import json
import requests
import os
import regex as re

from os.path import join, dirname, isfile
path = dirname(__file__)

with open(join(path, 'news.json'), 'r') as f:
    data = json.load(f)


def should_include_word(word):
    return len(word) >= min_len

min_len = 4

def parse_article(article):
    text = article['title'] + article.get('description', '')
    text = re.sub("\p{P}+", "", text).lower()
    text = text.split(' ')
    text = filter(should_include_word, text)
    return list(text)


parsed_data = {}
for date in data:
    articles = data[date]['articles']
    parsed_data[date] = [parse_article(a) for a in articles]


with open(join(path, 'news_text.json'), 'w') as f:
    json.dump(parsed_data, f, ensure_ascii=False)
