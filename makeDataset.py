import pandas as pd
import json

columns = ['user', 'text']

dataset = pd.DataFrame(columns=columns)

search_trump = pd.read_csv('data/realDonaldTrump.csv')

search_trump = search_trump.drop(columns=['id', 'date', 'Unnamed: 0'])

mega_trump_data = []

with open('data/realdonaldtrump.ndjson') as f:
    for line in f:
        tweet = json.loads(line)
        if tweet['text'][:3] == 'RT ':
            continue
        user = '@' + tweet['user']['screen_name']
        text = tweet['text']

        mega_trump_data.append([user, text])

mega_trump_df = pd.DataFrame(mega_trump_data, columns = columns)

search_obama = pd.read_csv('data/BarackObama.csv')

search_obama = search_obama.drop(columns=['id', 'date', 'Unnamed: 0'])

tweets = pd.read_csv('data/tweets.csv')

tweets = tweets[9450:9816]

tweets = tweets.drop(columns=['country', 'date_time', 'id', 'language', 'latitude',
                              'longitude', 'number_of_likes', 'number_of_shares'])

tweets = tweets.rename(columns={'author': 'user', 'content': 'text'})

tweets.user = '@BarackObama'

search_sanders = pd.read_csv('data/BernieSanders.csv')

search_sanders = search_sanders.drop(columns=['id', 'date', 'Unnamed: 0'])

dataset = dataset.append(search_trump, ignore_index=True)

dataset = dataset.append(mega_trump_df, ignore_index=True)

dataset = dataset.append(search_obama, ignore_index=True)

dataset = dataset.append(tweets, ignore_index=True)

dataset = dataset.append(search_sanders, ignore_index=True)

print(dataset.groupby('user').count())

dataset.to_csv('data/dataset.csv')
