from sklearn import svm, metrics
import cmapss_data as data
import threading
from queue import Queue
import numpy as np
import pickle
import multiprocessing

max_RUL = 90
X_train, y_train = data.get_train_data('train_FD001.csv', maximum_RUL=max_RUL)
X_valid, y_valid = data.get_valid_test_data('valid_FD001.csv', 'valid_RUL_FD001.csv', maximum_RUL=max_RUL)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std

hyper_param_list = []
data_read_lock = threading.Lock()
hyper_param_list_lock = threading.Lock()
print_lock = threading.Lock()
iter = 0

q = Queue()


def calc_svr(C, gamma):
    global iter, hyper_param_list

    clf = svm.SVR(kernel='rbf', C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    train_mae = metrics.mean_absolute_error(y_train, clf.predict(X_train))
    train_mse = metrics.mean_squared_error(y_train, clf.predict(X_train))
    valid_mae = metrics.mean_absolute_error(y_valid, clf.predict(X_valid))
    valid_mse = metrics.mean_squared_error(y_valid, clf.predict(X_valid))

    with hyper_param_list_lock:
        hyper_param_list.append((C, gamma, max_RUL, train_mae, train_mse, valid_mae, valid_mse))

    with print_lock:
        iter += 1
        print('Iteration: ' + str(iter))
        print('C: ' + str(C))
        print('gamma: ' + str(gamma))
        print('max RUL: ' + str(max_RUL))
        print('\t' + 'Train Mean Absolute Error: ' + str(train_mae))
        print('\t' + 'Train Mean Squared Error: ' + str(train_mse))
        print('\t' + 'Validation Mean Absolute Error: ' + str(valid_mae))
        print('\t' + 'Validation Mean Squared Error: ' + str(valid_mse))
        print('\n')

def threader():
    while True:
        C, gamma = q.get()
        calc_svr(C, gamma)
        q.task_done()

#Alle Threads starten
for i in range(multiprocessing.cpu_count()):
    t = threading.Thread(target=threader)
    t.daemon = True
    t.start()

C_vals = np.linspace(start=2**-5, stop=2**4, num=10)
gamma_vals = np.linspace(start=2**-5, stop=2**4, num=10)
for i in range(len(C_vals)):
    for j in range(len(gamma_vals)):
        C = C_vals[i]
        gamma = gamma_vals[j]
        q.put((C,gamma))


#Auf alle Threads warten
q.join()

lowest_valid_mse = 999999
best_hyper_param = None
for item in hyper_param_list:
    if item[6] < lowest_valid_mse:
        lowest_valid_mse = item[6]
        best_hyper_param = item

print(best_hyper_param)

clf = svm.SVR(kernel='rbf', C=best_hyper_param[0], gamma=best_hyper_param[1])
clf.fit(X_train, y_train)
pickle.dump(clf, open('pickles/SVR_' + str(max_RUL) + '_max_RUL.p', 'wb'))