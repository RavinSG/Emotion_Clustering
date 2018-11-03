from pymongo import MongoClient
import csv


def add_tweets(file_name):
    client = MongoClient('localhost', 27017)
    db = client['InsightsDB']

    corpus_DB = db.corpus
    tweet_DB = db.tweets

    tweet_list = []

    with open(file_name, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = True  # Skip header
        for row in reader:
            if header:
                header = False
                continue
            tweet_object = {}
            tweet_object['id'] = row[0]
            tweet_object['tweet'] = row[1]
            inserted_tweet = tweet_DB.insert_one(tweet_object)
            tweet_list.append(inserted_tweet.inserted_id)

    corpus = {'id': '1', 'name': 'Tweets Set 1', 'tweets': tweet_list[:100]}
    corpus_DB.insert_one(corpus)

    corpus = {'id': '2', 'name': 'Tweets Set 2', 'tweets': tweet_list[100:]}
    corpus_DB.insert_one(corpus)



if __name__ == '__main__':
    add_tweets('assets\\sample_data.csv')
