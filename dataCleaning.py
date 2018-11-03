import numpy as np
import pandas as pd
from tklearn.datasets.load_ait import load_ait

tweet_emotions = load_ait('H:\Research\\tkresources\datasets\SemEval-2018')[0].iloc[:, 2: -1]
file = np.load("X.npy")
data = pd.DataFrame(file)
data = data.join(tweet_emotions)

print(data)
data.to_csv('tweet_emotions.csv')
