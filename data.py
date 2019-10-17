from datetime import datetime, timedelta
import numpy as np
import json
import requests
import os

from os.path import join, dirname, isfile
path = dirname(__file__)

from newsapi import NewsApiClient

with open(join(path, '.key'), 'r') as f:
  newsapi_key = f.readlines()[0]
  

sources = 'abc-news, al-jazeera-english, associated-press, bbc, cnn, fox-news, msnbc, nbc-news, politico, reuters, the-hill, the-huffington-post, the-wall-street-journal, the-new-york-times, the-washington-post, the-washington-times, time, usa-today'
sort_by = 'popularity'

api = NewsApiClient(api_key=newsapi_key)

days_back = 29 # limit.
page_size = 100 # default 20, up to 100

QUERIES = [
  ('politics', 'news.json'),
  ('politics china', 'china.json'),
  ('politics russia', 'russia.json'),
  ('politics election', 'election.json'),
  ('politics turkey', 'turkey.json'),
  ('politics democrat', 'democrat.json'),
  ('politics republican', 'republican.json'),
  ('politics saudi', 'saudi.json'),
  ('politics trump', 'trump.json'),
  ('politics medicare', 'medicare.json'),
  ('politics brexit', 'brexit.json'),
]

for query, outname in QUERIES:
  jsonpath = join(path, f"raw_data/{outname}")

  if isfile(jsonpath):
      with open(jsonpath, 'r') as f:
          headline_data = json.load(f)
  else:
      headline_data = {}

  current_date = datetime.now() - timedelta(days=days_back)
  match = datetime.now().strftime('%Y-%m-%d')

  while current_date.strftime('%Y-%m-%d') != match:
      day_string = current_date.strftime('%Y-%m-%d')
      if day_string in headline_data:
          current_date += timedelta(days=1)
          continue
      headline_data[day_string] = api.get_everything(q=query,
                                     language='en',
                                     page_size=page_size,
                                     sources=sources,
                                     sort_by=sort_by,
                                     from_param=day_string,
                                     to=day_string)
      print(day_string)
      current_date += timedelta(days=1)

  with open(jsonpath, 'w') as f:
      json.dump(headline_data, f)
