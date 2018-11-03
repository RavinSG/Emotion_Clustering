from tklearn.datasets.load_ait import load_ait
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))
from insights.tasks.featurizer import featurize
import os
from config import GLOBAL_RESOURCE_PATH
import pickle

# Dataset path
DATASETS = {
    'emoint': os.path.join(GLOBAL_RESOURCE_PATH, 'datasets', 'EmoInt-2017'),
    'aitec': os.path.join(GLOBAL_RESOURCE_PATH, 'datasets', 'SemEval-2018'),
    'iest': os.path.join(GLOBAL_RESOURCE_PATH, 'datasets', 'IEST-2018'),
    'sentiment140': os.path.join(GLOBAL_RESOURCE_PATH, 'datasets', 'Sentiment140', 'training.1600000.processed.noemoticon.csv'),
}

train, dev, test = load_ait(DATASETS['aitec'], 'E.c')

X = featurize('Negations', train['Tweet'])

if hasattr(X, 'todense'):
    X = X.todense()
if hasattr(X, 'tolist'):
    X = X.tolist()

with open(r"../var/{}.{}.pickle".format('aitec', 'Negations'), "wb") as output_file:
    pickle.dump(X, output_file)

for i in X:
    print(i)
print(len(X))