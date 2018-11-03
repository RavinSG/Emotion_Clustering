import os

if 'TKLEARN_RESOURCES' in os.environ:
    GLOBAL_RESOURCE_PATH = os.environ['TKLEARN_RESOURCES']
else:
    GLOBAL_RESOURCE_PATH = 'H:\Research\\tkresources'

# Path to external models
PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

TWITTER_W2V_PATH = os.path.join(
    GLOBAL_RESOURCE_PATH, 'models', 'twitter.word2vec', 'word2vec_twitter_model.bin')
GOOGLE_W2V_PATH = os.path.join(
    GLOBAL_RESOURCE_PATH, 'models', 'google.word2vec', 'GoogleNews-vectors-negative300.bin')
GOOGLE_E2V_PATH = os.path.join(
    GLOBAL_RESOURCE_PATH, 'models', 'emoji2vec', 'emoji2vec.txt.gz')
GLOVE_PATH = os.path.join(GLOBAL_RESOURCE_PATH, 'models',
                          'glove.twitter.27B', 'glove.twitter.27B.200d.txt')
WIKI_FT_PATH = os.path.join(
    GLOBAL_RESOURCE_PATH, 'models', 'fastText', 'wiki-news-300d-1M.vec.gz')
WIKIS_FT_PATH = os.path.join(
    GLOBAL_RESOURCE_PATH, 'models', 'fastText', 'wiki-news-300d-1M-subword.vec.gz')
DEEP_MOJI_PATH = os.path.join(GLOBAL_RESOURCE_PATH, 'models', 'DeepMoji')
TWEET_NLP_PATH = os.path.join(
    GLOBAL_RESOURCE_PATH, 'models', 'TweetNLP', 'ark-tweet-nlp-0.3.2.jar')
DOC2VEC_PATH = os.path.join(
    GLOBAL_RESOURCE_PATH, 'models', 'doc2vec', 'releases_doc2vec.model')

# DATA PATHS
INPUT_PATHS = {
    'emoint': os.path.join(GLOBAL_RESOURCE_PATH, 'datasets', 'EmoInt-2017'),
    'aitec': os.path.join(GLOBAL_RESOURCE_PATH, 'datasets', 'SemEval-2018'),
    'iest': os.path.join(GLOBAL_RESOURCE_PATH, 'datasets', 'IEST-2018'),
    'iest_vad': os.path.join(GLOBAL_RESOURCE_PATH, 'datasets', 'IEST-2018'),
}

TEMP_PATH = os.path.join(PROJECT_PATH, 'temp')

# Feature lists
EMOINT_FEATURES = ['mpqa', 'bing_liu', 'affinn', 'sentiment140', 'nrc_hashtag_score', 'nrc_exp_emotion', 'nrc_hashtag',
                   'senti_wordnet', 'nrc_emotion', 'neg']


SERVER_NAME = None
MONGO_HOST = 'localhost'
MONGO_PORT = 27017

MONGO_DBNAME = 'InsightsDB'
