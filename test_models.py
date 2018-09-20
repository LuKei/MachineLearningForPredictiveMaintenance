from sklearn import svm, metrics
import cmapss_data as data
import threading
from queue import Queue
import numpy as np
import pickle
import pandas as pd

max_RUL = 9999
X_train, X_valid, X_test, y_train, y_valid, y_test = data.get_data_train_valid('train_FD001.csv', 'valid_FD001.csv',
                                                                                'valid_RUL_FD001.csv', 'test_FD001.csv',
                                                                                'test_RUL_FD001.csv', maximum_RUL=max_RUL)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

clf = pickle.load(open('pickles/SVR_110_max_RUL.p', 'rb'))
test_mae = metrics.mean_absolute_error(y_train, clf.predict(X_train))
test_mse = metrics.mean_squared_error(y_train, clf.predict(X_train))
print('\t' + 'SVR Test Mean Absolute Error: ' + str(test_mae))
print('\t' + 'SVR Mean Squared Error: ' + str(test_mse))

clf = pickle.load(open('pickles/MLP_100_max_RUL.p', 'rb'))
test_mae = metrics.mean_absolute_error(y_train, clf.predict(X_train))
test_mse = metrics.mean_squared_error(y_train, clf.predict(X_train))
print('\t' + 'MLP Test Mean Absolute Error: ' + str(test_mae))
print('\t' + 'MLP Mean Squared Error: ' + str(test_mse))