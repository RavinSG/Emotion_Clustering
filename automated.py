import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import pickle
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from sklearn import preprocessing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_data(file_path):
    tweets = pd.read_csv(file_path)
    tweets.set_index('Tweet ID', inplace=True)
    word2Vec_features = tweets.iloc[:, 0:300]
    labels = tweets.iloc[:, 300:]
    labels = np.array(labels.values.tolist())
    label_columns = tweets.columns.values[-10:]
    return word2Vec_features, labels


def use_PCA(features, no_dim):
    pca = PCA(n_components=no_dim)
    principal_components = pca.fit_transform(features)
    principalDF = pd.DataFrame(data=principal_components)
    return principalDF


def get_cluster_labels(label_set):
    cluster_labels = []
    for label in label_set:
        cluster_label = []
        for emotion in range(len(label)):
            if label[emotion] == 1:
                cluster_label.append(emotion)
        if len(cluster_label) > 0:
            cluster_labels.append(cluster_label[0])
        else:
            cluster_labels.append(-1)
    return np.array(cluster_labels)


def KMeans_cluster(data):
    features = use_PCA(data, 8)
    kMeans_model = KMeans(n_clusters=10).fit(features)
    k_labels = kMeans_model.labels_
    print(metrics.silhouette_score(features, k_labels))
    # pprint.pprint(get_cluster_labels(labels))
    return k_labels


def match_clusters(cluster_labels, emotion_labels):
    clusters = {}
    for i in range(len(cluster_labels)):
        if cluster_labels[i] not in clusters:
            clusters[cluster_labels[i]] = [0]
            clusters[cluster_labels[i]].append(emotion_labels[i])
        else:
            clusters[cluster_labels[i]].append(emotion_labels[i])
    return clusters


def rename_columns(df):
    df.columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']
    df.drop(['negative', 'positive'],1, inplace=True)
    return (df)


features, all_labels = get_data('tweet_emotions.csv')
emotion_labels = get_cluster_labels(all_labels)

with open('var/aitec.NRCAI.pickle','rb') as f:
    X = pickle.load(f)

X = pd.DataFrame(X)
# X = rename_columns(X)
print(X)
cluster_labels = KMeans_cluster(X)
print(cluster_labels)
print(emotion_labels)

print("homogeneity_score : ", homogeneity_score(emotion_labels, cluster_labels))
print("completeness_score : ", completeness_score(emotion_labels, cluster_labels))
print("v_measure_score : ", v_measure_score(emotion_labels, cluster_labels))

