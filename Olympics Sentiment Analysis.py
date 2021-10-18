#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tweepy


# In[2]:


# import tweepy
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

import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# your Twitter API key and API secret
my_api_key = 'Tcpy0hxzZSuyNzo9FY7qeId8q'
my_api_secret = 'wj5XMRGWwznRQpI1IN9HrGC3wYjQYHRSrfVAWjYTRUzIKEO6ca'

# authenticate
auth = tw.OAuthHandler(my_api_key, my_api_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# In[46]:


# set search filter for tweets - keywords and filter out retweets
search_query = "naomi+osaka -filter:retweets"


# In[47]:


# get tweets from the API
tweets = tw.Cursor(api.search_tweets,
              q=search_query,
              lang="en",
              ).items(200)

# store the API responses in a list
tweets_copy = []
for tweet in tweets:
    tweets_copy.append(tweet)
    
print("Total Tweets fetched:", len(tweets_copy))


# In[49]:


# intialize the dataframe
tweets_df = pd.DataFrame()

# populate the dataframe
for tweet in tweets_copy:
    hashtags = []
    try:
        for hashtag in tweet.entities["hashtags"]:
            hashtags.append(hashtag["text"])
        text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
    except:
        pass
    tweets_df = tweets_df.append(pd.DataFrame({'user_name': tweet.user.name, 
                                               'user_location': tweet.user.location,\
                                               'user_description': tweet.user.description,
                                               'user_verified': tweet.user.verified,
                                               'date': tweet.created_at,
                                               'text': text, 
                                               'hashtags': [hashtags if hashtags else None],
                                               'source': tweet.source}))
    tweets_df = tweets_df.reset_index(drop=True)

# show the dataframe
tweets_df.head()


# In[50]:


all_tweets = [tweet.text for tweet in tweets_copy]
all_tweets


# In[51]:


def remove_url(txt):
    """Replace URLs found in a text string with nothing 
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with url's removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


# In[52]:


all_tweets_no_urls = [remove_url(tweet) for tweet in all_tweets]
all_tweets_no_urls[0:10000]


# In[53]:


# Create a list of lists containing lowercase words for each tweet
words_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_urls]
words_in_tweet[0:10000]


# In[54]:


# remove stop words
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# View a few words from the set
list(stop_words)[0:10000]


# In[55]:


# Remove stop words from each tweet list of words
tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]

tweets_nsw[0:10000]


# In[56]:


collection_words = []

tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in tweets_nsw]

tweets_nsw_nc[0:10000]


# In[16]:


get_ipython().system('pip install afinn')


# In[17]:


# load AFINN for sentiment analysis 
from afinn import Afinn
import pandas as pd


# In[18]:


afn=Afinn(language='en')


# In[57]:


all_words_nsw = list(itertools.chain(*tweets_nsw_nc))
all_words_nsw


# In[58]:


clean_text_df=all_words_nsw[0:10000]
clean_text_df


# In[59]:


# create dataframe for AFINN sentiment analysis and eliminate words with score zero
scores=[afn.score(article) for article in clean_text_df]
sentiment=['positive' if score > 0
          else 'negative' if score <0
              else 'neutral'
                  for score in scores]

sentiment_df=pd.DataFrame()
sentiment_df['topic']=clean_text_df
sentiment_df['scores']=scores
sentiment_df['sentiment']=sentiment
sentiment_df = sentiment_df[sentiment_df.scores != 0]
print(sentiment_df)


# In[60]:


# count number of each sentiment score for weighted average
sentiment_count=sentiment_df.groupby(['scores','sentiment'])['topic'].count()
print(sentiment_count)


# In[61]:


# write to csv file
sentiment_df.groupby(['scores','sentiment'])['topic'].count().to_csv('C:/Users/jenni/osaka_tweet_sentiments.csv',index=True)


# In[62]:


# find most common words
clean_text_df = list(itertools.chain(*tweets_nsw_nc))

counts_nsw = collections.Counter(all_words_nsw)

counts_nsw.most_common(50)


# In[ ]:




