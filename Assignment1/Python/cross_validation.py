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

clf_linear = svm.SVC(kernel='linear', C=1)
clf_poly = svm.SVC(kernel='poly', C=1)



# Test/train with linear and polynomial

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.4, random_state=0)

clf_linear.fit(X_train, y_train)	
predictions_linear = clf_linear.predict(X_test)
scores_linear = clf_linear.score(X_test,y_test)


clf_poly.fit(X_train, y_train)
predictions_poly = clf_poly.predict(X_test)
scores_poly = clf_poly.score(X_test, y_test)


print("Test/Train\n")

print("Accuracy linear: %0.3f (+/- %0.3f)" % (scores_linear.mean(), scores_linear.std() * 2))
print("Accuracy polynomial: %0.3f (+/- %0.3f)\n" % (scores_poly.mean(), scores_poly.std() * 2))



# Cross validation

pred_cross_linear = cross_val_predict(clf_linear, data, target, cv=5)
pred_cross_poly = cross_val_predict(clf_poly, data, target, cv=5)

scores_cross_linear = cross_val_score(clf_linear, data, target, cv=5)
scores_cross_poly = cross_val_score(clf_poly, data, target, cv=5)


print("Cross Validation\n")
print("Accuracy linear: %0.3f (+/- %0.3f)" % (scores_cross_linear.mean(), scores_cross_linear.std() * 2))
print("Accuracy polynomial: %0.3f (+/- %0.3f)\n" % (scores_cross_poly.mean(), scores_cross_poly.std() * 2))