from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
import numpy as np



iris = datasets.load_iris()

X = iris.data

y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(iris["data"], iris["target"])

X_new = np.array(  [[ 5.1,  3.5,  1.4,  0.2],
                    [ 4.9,  3. ,  1.4,  0.2],
                    [ 4.7,  3.2,  1.3,  0.2],
                    [ 4.6,  3.1,  1.5,  0.2]])

                    
prediction = knn.predict(X_new)

print(prediction)