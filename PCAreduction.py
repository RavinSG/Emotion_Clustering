import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

X = np.array([[-1, -1, 4], [-2, -1, 5], [-3, -2, 3],
              [1, 1, -1], [2, 1, -3], [3, 2, 4]])

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
principalDF = pd.DataFrame(data = principal_components
             , columns = ['principal component 1', 'principal component 2'])

print(principalDF)