import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import math

# Predicts if somebody followed a information retrieval course based on the numbers of neighbours and a random number.


# Load in and handle data 

df = pd.read_csv("../Data/ODI-2018.csv")

data = df.as_matrix(columns=["Number of neighbors sitting around you?",  "Give a random number"])
target = np.ravel(df.loc[:, ["Have you taken a course on information retrieval?"]])

good = []

# Some numbers entered where to high for scikit to handle
for i in range(len(data)):
	try:
		data[i][0] = float (data[i][0])/50
		data[i][1] = float (data[i][1])/50
		if data[i][0] and data[i][1] < 100:
			good.append(i)
	except ValueError:
		continue

# First line were nans
good.pop(0)

data = data[good,:]
target = target[good]

known = []

# Only three unknowns, not usefull for cross-validation
for i in range(len(target)):
	try:
		target[i] = int(target[i])
		known.append(i)
	except ValueError:
		continue

data = data[known,:]
target = target[known]

# Make target of type int (which they already were?)
target=target.astype('int')



# Setup the models

clf_linear = svm.SVC(kernel='linear', C=10)
clf_poly = svm.SVC(kernel='poly', C=10)



# Test/train with linear and polynomial

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.4, random_state=0)

clf_linear.fit(X_train, y_train)	
predictions_linear = clf_linear.predict(X_test)

clf_poly.fit(X_train, y_train)
predictions_poly = clf_poly.predict(X_test)

accuracy_linear = metrics.accuracy_score(y_test, predictions_linear)
accuracy_poly = metrics.accuracy_score(y_test, predictions_poly)

print("Test/Train:\nlinear: ", accuracy_linear, "polynomial: ", accuracy_poly)



# Cross validation

pred_cross_linear = cross_val_predict(clf_linear, data, target, cv=5)
pred_cross_poly = cross_val_predict(clf_linear, data, target, cv=5)

accuracy_linear_cross = metrics.accuracy_score(target, pred_cross_linear)
accuracy_poly_cross = metrics.accuracy_score(target, pred_cross_poly)

print("Cross Validation\nlinear: ", accuracy_linear_cross, "polynomial: ", accuracy_poly_cross)




