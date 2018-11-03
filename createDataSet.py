import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import preprocessing
import pickle
import re
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def use_pca(features, no_dim):
    pca = PCA(n_components=no_dim)
    principal_components = pca.fit_transform(features)
    principal_df = pd.DataFrame(data=principal_components)
    return principal_df


def use_svd(features, no_dim):
    svd = TruncatedSVD(n_components=no_dim)
    single_values = svd.fit_transform(features)
    dataframe = pd.DataFrame(data=single_values)
    return dataframe


def get_word2vec():
    tweets = pd.read_csv('tweet_emotions.csv')
    tweets.set_index('Tweet ID', inplace=True)
    word2vec_features = tweets.iloc[:, 0:300]
    word2vec_features = use_pca(word2vec_features, 15)
    labels = tweets.iloc[:, 300:]
    return word2vec_features, labels


def get_NRCAI():
    with open('var/aitec.NRCAI.pickle', 'rb') as f:
        tweets = pickle.load(f)
    df = pd.DataFrame(tweets)
    df.columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise',
                  'trust']
    df.drop(['anticipation', 'disgust', 'negative', 'positive', 'surprise', 'trust'], 1, inplace=True)
    return df


def reduce_unigram(n_dimensions):
    print("Opening file...")
    with open('var/aitec.unigram.pickle', 'rb') as f:
        tweets = pickle.load(f)
    print("Reducing dimensions....")
    n_features = use_svd(tweets, 100)
    print("Writing to pickle..")
    with open(r"var/{}.{}.{}.pickle".format('aitec', 'unigram', n_dimensions), "wb") as output_file:
        pickle.dump(n_features, output_file)
    return n_features


def get_unigram():
    with open('var/aitec.unigram.100.pickle', 'rb') as f:
        tweets = pickle.load(f)
    return tweets


def get_SentiWordNet():
    with open('var/aitec.SentiWordNet.pickle', 'rb') as f:
        tweets = pickle.load(f)
        tweets = pd.DataFrame(tweets)
    return tweets


def get_negation():
    with open('var/aitec.Negations.pickle', 'rb') as f:
        tweets = pickle.load(f)
        tweets = pd.DataFrame(tweets)
    return tweets


def get_emoticon():
    with open('var/aitec.Emoticon.pickle', 'rb') as f:
        tweets = pickle.load(f)
        tweets = pd.DataFrame(tweets)
    return tweets


def get_data(word2vec=True, nrcai=True, unigram=True, sentiwordnet=True, negation=True,
             emoticon=True, word2vec_pca=10, unigram_pca=100, scale=True):
    df = pd.DataFrame
    word2vec_features, labels = get_word2vec()

    if word2vec:
        df = use_pca(word2vec_features, word2vec_pca)
    if sentiwordnet:
        sentiment = get_SentiWordNet()
        if df.empty:
            df = sentiment
        else:
            df = pd.concat([df, sentiment], 1)
    if nrcai:
        NRCAI_data = get_NRCAI()
        if df.empty:
            df = NRCAI_data
        else:
            df = pd.concat([df, NRCAI_data], 1)
    if unigram:
        unigram_data = use_pca(get_unigram(), unigram_pca)
        if df.empty:
            df = unigram_data
        else:
            df = pd.concat([df, unigram_data], 1)
    if negation:
        negation_data = get_negation()
        if df.empty:
            df = negation_data
        else:
            df = pd.concat([df, negation_data], 1)
    if emoticon:
        emoticon_data = get_emoticon()
        if df.empty:
            df = emoticon_data
        else:
            df = pd.concat([df, emoticon_data], 1)
    if scale:
        if not df.empty:
            df = preprocessing.scale(df)
            df = pd.DataFrame(df)

    return df, labels


def get_tweets():
    file = open('var/2018-E-c-En-train.txt', 'r', encoding='utf8')
    tweets = []
    for i in file:
        tweets.append(re.split(r'\t+', i)[1])
    tweets = tweets[1:]
    return tweets
