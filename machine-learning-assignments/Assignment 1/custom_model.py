from sklearn import linear_model
from analysis_and_preprocessing import main
import numpy as np

def model_evaluation(X_train, y_train):
    # input: X_train and y_train matrices
    # output: LinearRegression model's coef_ and intercept_ parameters
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    coef_ = (regr.coef_)
    intercept_ = (np.array([regr.intercept_]))
    return coef_, intercept_

def predict(instance, coef_, intercept_):
    # input: instance matrix, coef_ array and intercept_ array
    # ouput: list of predictions for input instances
    regr = linear_model.LinearRegression()
    regr.coef_ = coef_
    regr.intercept_ = intercept_
    predictions = regr.predict(instance)
    return predictions