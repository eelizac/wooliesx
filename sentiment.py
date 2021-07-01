"""
    This document employs Tweepy to extract twitter search data. 
    It also applies textblob to get a sentiment analysis and creates
    a data frame and histogram. 
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

import tweepy as tw
import nltk
from nltk.corpus import stopwords
import re
import networkx
from textblob import TextBlob
from helper import remove_url

import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# tokens 
consumer_key=''
consumer_secret=''
access_token=''
access_token_secret=''

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Create a custom search term and define the number of tweets
search_term = 'dilmah tea-filter:retweets'

tweets = tw.Cursor(api.search, q=search_term, lang="en", since='2020-11-01').items(1000)

# Remove URLs
tweets_no_urls = [remove_url(tweet.text) for tweet in tweets]

# Analyse sentiments in tweet --> textblob calculates the polarity values of individual tweets 
# create textblob objects of the tweet 
sentiment_objects = [TextBlob(tweet) for tweet in tweets_no_urls]

# create list of polarity values and tweet texts 
sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]

# create a dataframe containing the polairty value and tweet text 
sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "tweet"])

# remove polarity values equal to 0 
sentiment_df = sentiment_df[sentiment_df.polarity != 0]
# plot dataframe into a histogram to highlight overall sentiment from skew. 
fig, ax = plt.subplots(figsize=(8,6))

# plot histogram of the polarity values
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], ax=ax, color = "purple")

plt.title("Sentiments from Tweets on Dilmah Tea")
plt.show()
#print(sentiment_df)
#sentiment_df.to_csv(r'/import/kamen/4/z5312689/Desktop/sentiment.csv')
