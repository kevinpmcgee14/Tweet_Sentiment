import os
import requests
from requests.auth import AuthBase
from requests.auth import HTTPBasicAuth
from urllib.parse import quote_plus, quote
from datetime import datetime, timedelta
from base64 import b64encode

class TwitterAuth(AuthBase):
  def __init__(self):
    self.bearer_token_url = "https://api.twitter.com/oauth2/token"
    self.consumer_key = os.getenv('TWITTER_API_KEY')
    self.consumer_secret = os.getenv('TWITTER_API_SECRET')
    self.bearer_token = self.get_bearer_token()

  def get_bearer_token(self):
    encoded_auth = quote(self.consumer_key) + ':' +  quote(self.consumer_secret)
    auth_header =  'Basic ' + b64encode(encoded_auth.encode('ascii')).decode('ascii')
    response = requests.post(
      self.bearer_token_url, 
      data={'grant_type': 'client_credentials'},
      headers={
        'Authorization': auth_header,
        'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
      })

    if response.status_code is not 200:
      raise Exception(f"Cannot get a Bearer token (HTTP %d): %s" % (response.status_code, response.text))

    body = response.json()
    return body['access_token']

  def __call__(self, r):
    r.headers['Authorization'] = f"Bearer %s" % self.bearer_token
    r.headers['User-Agent'] = 'TwitterDevFilteredStreamQuickStartPython'
    return r

class TwitterApp():

  def __init__(self):
    self.auth = TwitterAuth()

  def get(self, endpoint):
    return requests.get(endpoint, auth=self.auth)

  def get_hashtag(self, hashtag: str, result_type: str, date_range: tuple):
      search_endpoint = 'https://api.twitter.com/1.1/search/tweets.json'

      final_response = []
      lowest_id = None
      min_date, max_date = date_range
      date = max_date
      before_date = max_date + timedelta(days=1)
      day_count = 0
      while date >= min_date:
        query = '?q=' + quote_plus(hashtag) + '&' + quote_plus(result_type)
        query += '&lang=en&count=100&sample=50'
        if day_count == 2:
          lowest_id = None
          before_date = before_date - timedelta(days=1)
          day_count = 0
        query += '&until={}'.format(before_date.strftime('%Y-%m-%d'))
        if lowest_id:
            query += '&max_id={}'.format(lowest_id)
        endpoint=search_endpoint + query
        response = self.get(endpoint)
        if response.status_code == 200:
            tweet_data = response.json()['statuses']
            lowest_id = tweet_data[-1]["id_str"]
            date = datetime.strptime(tweet_data[-1]["created_at"], "%a %b %d %H:%M:%S %z %Y").date()
            final_response += tweet_data
            day_count += 1
        else:
          break
      return final_response