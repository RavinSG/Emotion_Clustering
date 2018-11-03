#
#  Featurizer task functionality
#


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from tklearn.feature_extraction import TransferFeaturizer, EmbeddingTransformer, LexiconVectorizer
from tklearn.preprocessing import DictionaryTokenizer, TweetTokenizer
# from tklearn.text.tokens import build_vocabulary

import os
import time
import pymongo
from bson.objectid import ObjectId

from insights.cache import google_word2vec, context2vec, twitter_word2vec, twitter_glove, wiki_fasttext, sentiment140_doc2vec


class Echo:
    def __call__(self, *args, **kwargs):
        return args[0]


class SplitSelect:
    def __init__(self, delimiter, idx):
        self.delimiter = delimiter
        self.idx = idx

    def __call__(self, texts, *args, **kwargs):
        return map(lambda text: text.split(self.delimiter)[self.idx], texts)


class Doc2VecLookup:
    def __init__(self, doc2vec='sentiment140'):
        self.doc2vec = doc2vec

    def __call__(self, texts, *args, **kwargs):
        if self.doc2vec == 'sentiment140':
            doc2vec = sentiment140_doc2vec()
        else:
            doc2vec = sentiment140_doc2vec()
        return map(doc2vec.infer_vector, texts)


def build_vocabulary(texts=None, tokenizer=None, preprocess=None):
    """
    Builds vocabulary form given text(s) using provided tokenizer. Text pre-processing is performed prior to
    tokenizing.

    :param texts: Input text or list of texts.
    :param tokenizer: None or callable
    :param preprocess: None or callable
    :return: Vocabulary set
    """
    if texts is None:
        texts = []
    elif isinstance(texts, str):
        texts = [texts]
    if tokenizer is None:
        def tokenizer(ts):
            return [t.split(' ') for t in ts]
    if preprocess is None:
        def preprocess(ts):
            return ts
    vocab = set()
    for x in tokenizer(preprocess(texts)):
        vocab.update(x)
    return vocab

def featurize(feature, dataset):
    if 'google_word2vec' == feature:
        word_vectors = google_word2vec(verbose=True)
        print('Creaeting tok')
        tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
        print('building vocab')
        vocab = build_vocabulary(*dataset, tokenizer=tt)
        ea = EmbeddingTransformer(
            word_vectors, vocab, output='average', default='ignore')
        return make_pipeline(tt, ea).fit_transform(dataset)
    elif 'google_word2vec-emoji2vec' == feature:
        word_vectors = google_word2vec(emoji2vec=True)
        tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
        vocab = build_vocabulary(*dataset, tokenizer=tt)
        ea = EmbeddingTransformer(
            word_vectors, vocab, output='average', default='ignore')
        return make_pipeline(tt, ea).fit_transform(dataset)
    elif 'twitter_word2vec' == feature:
        word_vectors = twitter_word2vec()
        tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
        vocab = build_vocabulary(*dataset, tokenizer=tt)
        ea = EmbeddingTransformer(
            word_vectors, vocab, output='average', default='ignore')
        return make_pipeline(tt, ea).fit_transform(dataset)
    elif 'wiki_fasttext' == feature:
        word_vectors = wiki_fasttext()
        tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
        vocab = build_vocabulary(*dataset, tokenizer=tt)
        ea = EmbeddingTransformer(
            word_vectors, vocab, output='average', default='ignore')
        return make_pipeline(tt, ea).fit_transform(dataset)
    elif 'wiki_fasttext-subword' == feature:
        word_vectors = wiki_fasttext(subword=True)
        tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
        vocab = build_vocabulary(*dataset, tokenizer=tt)
        ea = EmbeddingTransformer(
            word_vectors, vocab, output='average', default='ignore')
        return make_pipeline(tt, ea).fit_transform(dataset)
    elif 'twitter_glove' == feature:
        word_vectors = twitter_glove()
        tt = DictionaryTokenizer(word_vectors.vocabulary, ignore_vocab=True)
        vocab = build_vocabulary(*dataset, tokenizer=tt)
        ea = EmbeddingTransformer(
            word_vectors, vocab, output='vector', default='ignore')
        return make_pipeline(tt, ea).fit_transform(dataset)
    elif 'context2vec' == feature:
        c2v = FunctionTransformer(context2vec, validate=False)
        return c2v.fit_transform(dataset)
    elif 'unigram' == feature:
        tt = TweetTokenizer()
        cv = CountVectorizer(preprocessor=Echo(), tokenizer=Echo())
        return make_pipeline(tt, cv).fit_transform(dataset)
    elif 'bigram' == feature:
        tt = TweetTokenizer()
        cv = CountVectorizer(preprocessor=Echo(),
                             tokenizer=Echo(), ngram_range=(2, 2))
        return make_pipeline(tt, cv).fit_transform(dataset)
    elif 'tf-idf' == feature:
        tt = TweetTokenizer()
        tfidf = TfidfVectorizer(
            preprocessor=lambda x: x, tokenizer=lambda x: x)
        return make_pipeline(tt, tfidf).fit_transform(dataset)
    elif LexiconVectorizer.has_filter(feature):
        atv = LexiconVectorizer(feature, True)
        return atv.fit_transform(dataset)
    else:
        return []


def featurizer_task(db, r):
    corpus_DB = db.corpus
    tweets_DB = db.tweets
    cid = r['corpus']
    c = corpus_DB.find_one({'_id': ObjectId(cid)})
    tids = c['tweets']
    tweets = []
    for tid in tids:
        print(tid)
        tweet = tweets_DB.find_one({'_id': tid})
        t = tweet['tweet']
        tweets.append(t)
    features = [[] for _ in range(len(tweets))]
    for feature in r['features']:
        print('start featuizin', flush=True)
        X = featurize(feature, tweets)
        print('ened featuizin', flush=True)
        if hasattr(X, 'todense'):
            X = X.todense()
        if hasattr(X, 'tolist'):
            X = X.tolist()
        for i, x in enumerate(X):
            features[i] += x
    return features
