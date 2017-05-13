print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split,learning_curve
import data_method as dat

# import some data to play with
data, labels = dat.classData()

# X = data[129:]
# y = labels[129:385,2]
# X = data[0:257]
# y = labels[0:257,1]
X = data
y = labels


X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state = 5)

h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 10  # SVM regularization parameter

print(":::::::::")
svc = svm.SVC(kernel='linear', decision_function_shape='ovo',C=C).fit(X_train, y_train)
print(svc.score(X_test,y_test))



poly_svc = svm.SVC(kernel='poly', decision_function_shape='ovo',degree=3, C=C).fit(X_train, y_train)
print(poly_svc.score(X_test,y_test))

# lin_svc = svm.LinearSVC(decision_function_shape='ovo',C=C).fit(X_train, y_train)
# print(lin_svc.score(X_test,y_test))


# rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
# print(rbf_svc.score(X_test,y_test))