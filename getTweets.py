#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Downloads all tweets from a given user.
Uses twitter.Api.GetUserTimeline to retreive the last 3,200 tweets from a user.
Twitter doesn't allow retreiving more tweets than this through the API, so we get
as many as possible.
"""

from __future__ import print_function

import json
import sys
import pandas as pd

import twitter


def get_tweets(api=None, screen_name=None):
    timeline = api.GetUserTimeline(screen_name=screen_name, count=100)
    earliest_tweet = min(timeline, key=lambda x: x.id).id
    print("getting tweets before:", earliest_tweet)

    while True:
        tweets = api.GetUserTimeline(
            screen_name=screen_name, max_id=earliest_tweet, count=100
        )
        new_earliest = min(tweets, key=lambda x: x.id).id
        earliest_date = min(tweets, key=lambda x: x.id).created_at

        if not tweets or new_earliest == earliest_tweet:
            break
        else:
            earliest_tweet = new_earliest
            print("getting tweets before:", earliest_date)
            timeline += tweets

    return timeline


if __name__ == "__main__":

    api = twitter.Api(consumer_key="EBjoCXRayoBLjAVWyw44WkmbF",
                      consumer_secret="MK79AAVdxN9XN0TfFmMR3spXvYscSkn9w53GKXjRrP9RYws43k",
                      access_token_key="4605684514-xWuZOVBiRKx3OmCVvoljLyQcd1ss1TjPQgvN6De",
                      access_token_secret="M4CJEDXW683jHs0YOeKOHrZgJbU8AfnfEBqrQ6yxLOwd8")

    screen_name = sys.argv[1]
    print(screen_name)
    timeline = get_tweets(api=api, screen_name=screen_name)

    idx = 2

    with open('data/'+screen_name+str(idx)+'.json', 'w+') as f:
        f.write('{\n\"tweets\": [')
        for tweet in timeline:
            if tweet.text[:3] == 'RT ':
                continue
            f.write(json.dumps(tweet._json))
            if tweet is not timeline[-1]:
                f.write(',')
            f.write('\n')
        f.write(']}')

    data = []
    for tweet in timeline:
        if tweet.text[:3] == 'RT ':
            continue
        user = '@' + tweet.user.screen_name
        id = tweet.id
        date = tweet.created_at
        text = tweet.text

        data.append([user, id, date, text])

    df = pd.DataFrame(data, columns=['user', 'id', 'date', 'text'])

    df.to_csv('data/'+screen_name+str(idx)+'.csv')