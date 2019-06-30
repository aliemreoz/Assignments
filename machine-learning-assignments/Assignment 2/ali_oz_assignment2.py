# -*- coding: utf-8 -*-
# Ali Emre Öz

### by default the variable name for confusion matrix was "confusion matrix" but I changed
### it as confusionmatrix in order not to encounter error.

### In SGDClassifier, there was a warning for DeprecationWarning because of something interested with n_iter value.
### I add some lines on importing part for not getting this warnings.
### Found it from stackoverflow.com/questions/879173/how-to-ignore-deprecation-warnings-in-python

import _pickle as pickle
from collections import Counter
import itertools
import warnings
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.linear_model import SGDClassifier


# PART a)Feature Extraction
def feature1(x):
    """This feature computes the proportion of black squares to the
       total number of squares in the grid.
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature1_value: type-float
       """

    count = Counter([item for array in x for item in array])
    green_sq = float(count[0])
    black_sq = float(count[1])
    feature1_value = (black_sq) / (black_sq + green_sq)
    return feature1_value
def feature2(x):
    """This feature computes the sum of the max of continuous black squares
       in each row
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature2_value: type-float
       """
    all_freq = []
    for i in x:
        freq = []
        for b, g in itertools.groupby(i):
            if b == 1:
                freq.append(sum(g))
        try:
            all_freq.append(max(freq))
        except:
            all_freq.append(float(0))
    feature2_value= float(sum(all_freq))
    return feature2_value


# PART b) Preparing Data
def part_b():
    train_positives = pickle.load(open('training_set_positives.p', 'rb'))
    train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))
    X = []
    y = []
    for i in range(len(train_positives)):
        X.append([feature1(x=train_positives[i]), feature2(x=train_positives[i])])
        y.append(1)
    for i in range(len(train_negatives)):
        X.append([feature1(x=train_negatives[i]), feature2(x=train_negatives[i])])
        y.append(0)
    X = np.array(X)
    y = np.array(y)
    return X, y


# PART c) Classification with SGDClassifier
def part_c(x):

    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """
    predicted_class = None
    alpha = 0.001
    random_state = 0
    sgd_clf = SGDClassifier(alpha=alpha, max_iter=20, random_state=random_state)
    X, y = part_b()
    sgd_clf.fit(X, y)
    test = [[feature1(x), feature2(x)]]
    prediction = sgd_clf.predict(test)
    if prediction == [0]:
        predicted_class = 0
    elif prediction == [1]:
        predicted_class = 1
    return predicted_class


# PART d) Assess the performance of the classifier in part c
def part_d():

    sgd_clf = SGDClassifier(alpha=0.001, max_iter=20, random_state=0)
    X, y = part_b()
    y_train_pred = cross_val_predict(sgd_clf, X, y, cv=3)
    confusionmatrix = confusion_matrix(y, y_train_pred)
    precision = precision_score(y, y_train_pred)
    recall = (recall_score(y, y_train_pred))
    return [precision, recall, confusionmatrix]


# PART e) Classification with RandomForestClassifier
def part_e(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """
    predicted_class = None
    forest_clf = RandomForestClassifier(random_state=0)
    X, y = part_b()
    forest_clf.fit(X, y)
    test = [[feature1(x), feature2(x)]]
    prediction = forest_clf.predict(test)
    if prediction == [0]:
        predicted_class = 0
    elif prediction == [1]:
        predicted_class = 1
    return predicted_class


# PART f) Assess the performance of the classifier in part e
def part_f():
    forest_clf = RandomForestClassifier(random_state=0)
    X, y = part_b()
    y_train_pred = cross_val_predict(forest_clf, X, y, cv=3)
    confusionmatrix = confusion_matrix(y, y_train_pred)
    precision = precision_score(y, y_train_pred)
    recall = (recall_score(y, y_train_pred))
    return [precision, recall, confusionmatrix]


# PART g) Your Own Classification Model
def custom_model(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """
    train_positives = pickle.load(open('training_set_positives.p', 'rb'))
    train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))

    #feature 1 - green square / total square
    def my_feature_1(a):
        countt = Counter([item for array in a for item in array])
        green_sq = float(countt[0])
        black_sq = float(countt[1])
        my_feature_1 = (green_sq) / (black_sq + green_sq)
        return my_feature_1

    #feature 2 - sum over all the columns of the maximum number of continuous black squares in each columns
    def my_feature_2(a):
        all_freq = []
        a = zip(*a)
        for i in a:
            freq = []
            for b, g in itertools.groupby(i):
                if b == 1:
                    freq.append(sum(g))
            try:
                all_freq.append(max(freq))
            except:
                all_freq.append(float(0))
        my_feature_2 = float(sum(all_freq))
        return my_feature_2

    def create_feature_matrix():
        X = []
        y = []
        for i in range(len(train_positives)):
            X.append([my_feature_1(a=train_positives[i]), my_feature_2(a=train_positives[i])])
            y.append(1)
        for i in range(len(train_negatives)):
            X.append([my_feature_1(a=train_negatives[i]), my_feature_2(a=train_negatives[i])])
            y.append(0)
        X = np.array(X)
        y = np.array(y)
        return X, y

    predicted_class = None

    alpha = 0.001
    random_state = 0
    max_iter = 20
    sgd_clf = SGDClassifier(alpha = alpha, max_iter = max_iter, random_state = random_state)
    X, y = create_feature_matrix()
    sgd_clf.fit(X, y)
    test = [[my_feature_1(x), my_feature_2(x)]]
    prediction = sgd_clf.predict(test)
    if prediction == [0]:
        predicted_class = 0
    elif prediction == [1]:
        predicted_class = 1

    # y_train_pred = cross_val_predict(sgd_clf, X, y, cv=3)
    # confusionmatrix = confusion_matrix(y, y_train_pred)
    # precision = precision_score(y, y_train_pred)
    # recall = (recall_score(y, y_train_pred))

    return predicted_class


######## TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST ##########
# train_positives = pickle.load(open('training_set_positives.p', 'rb'))
# train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))
#
# print("-"*50)
# print("part_a_vol_1")
# print(feature1(train_positives[0]))
# print("*"*50)
#
# print("part_a_vol_2")
# print(feature2(train_positives[2]))
# print("*"*50)
#
# print("part_b")
# X, y = part_b()
# for i in range(0,300,25):
#     print(X[i],y[i])
# print("*" * 50)
#
#
# print("part_c")
# for i in range(10):
#     print(part_c(train_positives[i]))
# print("-"*50)
# print("*" * 50)
#
# print("part_d")
# print(part_d())
# print("*" * 50)
#
# print("part_e")
# for i in range(10):
#     print(part_e(train_negatives[i]))
# print("*" * 50)
#
# print("part_f")
# print(part_f())
# print("*" * 50)
# print("custom_model")
# for i in range(10):
#     print(custom_model(train_negatives[i]))
# print("-"*50)
######## TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST ##########