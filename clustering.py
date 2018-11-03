import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import random
import os.path
import pprint
from sklearn import metrics, mixture
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from matplotlib import pyplot as plt
import createDataSet


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# Reduces the dimensions of the data set
def reduce_dimensions(feature_set, no_dim):
    pca = PCA(n_components=no_dim)
    principal_components = pca.fit_transform(feature_set)
    principal_df = pd.DataFrame(data=principal_components)
    return principal_df


# Takes the label array per tweets and return one emotion for every tweet
def get_cluster_labels(label_set, binary=False):
    cluster_labels = []
    if not binary:
        if not os.path.isfile('var/randomLabels.pickle'):
            for label in label_set:
                cluster_label = []
                for emotion in range(len(label)):
                    if label[emotion] == 1:
                        cluster_label.append(emotion)
                if len(cluster_label) > 0:
                    cluster_labels.append(random.choice(cluster_label))
                else:
                    cluster_labels.append(10)
            pd.to_pickle(cluster_labels, 'var/randomLabels.pickle')
        else:
            cluster_labels = pd.read_pickle('var/randomLabels.pickle')
    else:
        for label in label_set:
            semantic = 14
            positive = label * np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
            negative = label * np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 0])
            if sum(positive) > 0 and sum(negative) == 0:
                semantic = 11  # positive tweet
            elif sum(positive) == 0 and sum(negative) > 0:
                semantic = 12  # negative tweet
            elif sum(positive) > 0 and sum(negative) > 0:
                semantic = 13  # mixed tweet
            elif sum(positive) == sum(negative) == 0:
                semantic = 14  # no emotion
            cluster_labels.append(semantic)
    return np.array(cluster_labels)


# Cluster using KMeans
def use_kmeans(data, clusters=5):
    kmeans_model = KMeans(n_clusters=clusters).fit(data)
    k_labels = kmeans_model.labels_
    return k_labels


# Cluster using DBScan
def use_dbscan(data):
    db = DBSCAN(min_samples=23, eps=0.16).fit(data)
    db_labels = db.labels_
    label_types = set(db_labels)
    for i in label_types:
        print(i, ' : ', db_labels.tolist().count(i))
    return db.labels_


def use_gmm(data, clusters=4):
    clf = mixture.GaussianMixture(n_components=clusters, covariance_type='full')
    clf.fit(data)
    return clf.predict(data)


# Returns a dictionary which has cluster as the key and the tweets in
# the cluster as values
def segregate_clusters(cluster_labels, emotion_label):
    clusters = {}
    for i in range(len(cluster_labels)):
        if cluster_labels[i] not in clusters:
            clusters[cluster_labels[i]] = []
            clusters[cluster_labels[i]].append(emotion_label[i])
        else:
            clusters[cluster_labels[i]].append(emotion_label[i])
    return clusters


# Scoring for clustering
def get_scores(feature_set, emotion_label, cluster_labels):
    homogeneity = homogeneity_score(emotion_label, cluster_labels)
    completeness = completeness_score(emotion_label, cluster_labels)
    v_measure = v_measure_score(emotion_label, cluster_labels)
    silhouette = metrics.silhouette_score(feature_set, cluster_labels)
    return [homogeneity, completeness, v_measure, silhouette]


# Divide the whole data set into small batches
def make_batches(feature_set, data_labels, tweet_set, batch_size=100):
    batches = []
    batch_labels = []
    batch_tweets = []
    data_size = len(feature_set)
    start_point = 0
    end_point = batch_size

    while end_point <= data_size:
        batch = feature_set[start_point:end_point]
        batch_label = data_labels[start_point:end_point]
        tweet = tweet_set[start_point:end_point]
        batches.append(batch)
        batch_labels.append(batch_label)
        batch_tweets.append(tweet)
        start_point = end_point
        end_point = end_point + batch_size

    final_batch = feature_set[start_point:data_size]
    final_labels = data_labels[start_point:data_size]
    final_tweets = tweet_set[start_point:data_size]
    batches.append(final_batch)
    batch_labels.append(final_labels)
    batch_tweets.append(final_tweets)
    return batches, batch_labels, batch_tweets


# Draws the pie chart of the clusters
def visualize_clusters(clusters):
    charts = {}
    total_emotions = {}
    # fig = plt.figure()
    for i in range(len(clusters)):
        cluster = clusters[i]
        emotions = set(cluster)
        emotions_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness',
                          'surprise', 'None', 'positive', 'negative', 'mixed', 'dull']
        cluster_emotions = []
        counter = []
        for j in emotions:
            emotion_name = emotions_names[int(j)]
            cluster_emotions.append(emotion_name)
            num = cluster.count(j)
            counter.append(num)
            if emotion_name in total_emotions.keys():
                total_emotions[emotion_name] = total_emotions[emotion_name] + num
            else:
                total_emotions[emotion_name] = num
        print(sum(counter))
        charts[i] = [cluster_emotions, counter]
    axe = 0
    for k in charts:
        axis = plt.subplot2grid((2, len(charts)), (0, axe))
        bar = plt.subplot2grid((2, len(charts)), (1, axe))
        axis.pie(charts[k][1], labels=charts[k][0], startangle=90, autopct='%1.1f%%')
        percent = []
        tweet_no = 0
        for i in range(len(charts[k][1])):
            tweet_no = tweet_no + charts[k][1][i]
            val = charts[k][1][i]/total_emotions[charts[k][0][i]]
            percent.append(val)
        plt.title("Tweets: "+ str(tweet_no))
        bar.bar(charts[k][0], percent)
        bar.set_yticks([0, 0.25, 0.5, 0.75, 1])
        for label in bar.xaxis.get_ticklabels():
            label.set_rotation(90)
        axis.axis('equal')
        axe = axe + 1
    plt.suptitle("Features: Sentiwordnet, NRCAI, Negation  Algorithm: GMM\n Dimensions: 4 (Scaled)")
    plt.show()
    print(total_emotions)
    return charts


features, labels = createDataSet.get_data(word2vec=False, nrcai=True, unigram=False,
                                          sentiwordnet=True, negation=True, emoticon=False, scale=True)
tweets = createDataSet.get_tweets()
features = reduce_dimensions(features, 4)

# Pie chart of the whole data set
def visualize_data_set(data_labels):
    # data_labels = get_cluster_labels(data_labels)
    data_labels = data_labels.tolist()
    count = {}
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness',
                'surprise', 'None', 'positive', 'negative', 'mixed', 'dull']
    for i in range(len(emotions)):
        num = data_labels.count(i)
        if num > 0:
            count[emotions[i]] = num

    pprint.pprint(count)
    fig1, ax1 = plt.subplots()
    ax1.pie(count.values(), labels=count.keys(), autopct='%1.1f%%')
    ax1.axis('equal')
    plt.show()
    return None


def generate_clusters(feature_set, label_set, tweet_set, batch_size=500):
    v_score_total = 0
    silhouette_total = 0
    feature_batches, label_batches, tweet_batches = \
        make_batches(feature_set=feature_set, data_labels=label_set, tweet_set=tweet_set, batch_size=batch_size)
    for i in range(len(feature_batches)):
        batch_cluster_labels = use_gmm(feature_batches[i], clusters=5)
        scores = get_scores(feature_batches[i], label_batches[i], batch_cluster_labels)
        v_score_total = v_score_total + scores[2]
        silhouette_total = silhouette_total + scores[3]
    v_score_average = v_score_total / len(feature_batches)
    silhouette_average = silhouette_total / len(feature_batches)
    return v_score_average, silhouette_average


features = features.values.tolist()
labels = np.array(labels.values.tolist())
single_labels = get_cluster_labels(labels, binary=True)
print(set(single_labels))
# visualize_data_set(single_labels)
feature_batches, label_batches, tweet_batches = make_batches(np.array(features),
                                                             single_labels, tweets, batch_size=6000)
label_batches = pd.DataFrame(label_batches)
# print(label_batches.values.tolist())

cluster_labels = use_gmm(feature_batches[0], clusters=5)
tweet_labels = label_batches.values.tolist()[0]
print(cluster_labels)
seg_clusters = segregate_clusters(cluster_labels, tweet_labels)
pprint.pprint(visualize_clusters(seg_clusters))

# generate_clusters(features, single_labels, tweets)
score_list = []
for i in range(100,1100,100):
    v_score, silhouette_score = generate_clusters(features, single_labels, tweets, batch_size=i)
    score_list.append([i, v_score, silhouette_score])

scores = pd.DataFrame(score_list)
scores.columns = ['Batch size', 'v_score', 'silhouette_score']
scores.set_index('Batch size', inplace=True)
print(scores)
