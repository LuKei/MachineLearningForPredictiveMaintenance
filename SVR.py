from sklearn import svm
import cmapss_data as data
import threading
from queue import Queue
import numpy as np


X_train, X_test, y_train, y_test = data.get_sensor_data('train_FD001.csv','test_FD001.csv', 'RUL_FD001.csv')

hyper_param_list = []
hyper_param_list_lock = threading.Lock()
print_lock = threading.Lock()
iter = 0

q = Queue()


def calc_svr(C, gamma):
    global iter, hyper_param_list
    clf = svm.SVR(kernel='rbf', C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    prediction_RUL = clf.predict([X_test[0], X_test[1], X_test[2]])
    real_RUL = [y_test[0], y_test[1], y_test[2]]

    with hyper_param_list_lock:
        hyper_param_list.append((C, gamma, train_accuracy, test_accuracy))

    with print_lock:
        iter += 1
        print('Iteration: ' + str(iter))
        print('C: ' + str(C))
        print('C: ' + str(gamma))
        print('\t' + 'Train Accuracy: ' + str(train_accuracy))
        print('\t' + 'Test Accuracy: ' + str(test_accuracy))
        print('\t' + 'Prediction RUL: ' + str(prediction_RUL) + '\n')
        print('\t' + 'Real RUL: ' + str(real_RUL) + '\n')
        print('\n')

def threader():
    while True:
        C, gamma = q.get()
        calc_svr(C, gamma)
        q.task_done()

#Alle Threads starten
for i in range(12):
    t = threading.Thread(target=threader)
    t.daemon = True
    t.start()

C_vals = np.linspace(start=2**-5, stop=2**3, num=40)
gamma_vals = np.linspace(start=0.01, stop=2**3, num=40)
for i in range(len(C_vals)):
    for j in range(len(gamma_vals)):
        C = C_vals[i]
        gamma = gamma_vals[j]
        q.put((C,gamma))


#Auf alle Threads warten
q.join()

best_test_acc = -1
best_hyper_param = None
for item in hyper_param_list:
    if item[3] > best_test_acc:
        best_test_acc = item[3]
        best_hyper_param = item

print(best_hyper_param)



