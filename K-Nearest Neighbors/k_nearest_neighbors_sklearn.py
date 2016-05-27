import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

data_frame = pd.read_csv('breast-cancer-wisconsin-data.txt')

print('her')
# get rid of missing data and the 'id' column, which
# would completely throw off our algorithm
data_frame.replace(['?'], -99999, inplace=True)
data_frame.drop(['id'], 1, inplace=True)

# extract features and labels
X = np.array(data_frame.drop(['class'], 1))
y = np.array(data_frame['class'])

# shuffle data and create training / testing sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)
