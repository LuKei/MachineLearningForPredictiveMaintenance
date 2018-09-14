from sklearn import neural_network, metrics
import cmapss_data as data
import threading
from queue import Queue
import numpy as np


X_train, X_test, y_train, y_test = data.get_sensor_data('train_FD001_80split.csv','test_FD001_20split.csv',
                                                        'RUL_FD001_20split.csv', maximum_RUL=110)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

hyper_param_list = []
hyper_param_list_lock = threading.Lock()
print_lock = threading.Lock()
iter = 0

q = Queue()


def calc_mlp(neuron_count_1, neuron_count_2):
    global iter, hyper_param_list
    clf = neural_network.MLPRegressor(hidden_layer_sizes=(neuron_count_1, neuron_count_2), max_iter=500)
    clf.fit(X_train, y_train)
    train_mae = metrics.mean_absolute_error(y_train,clf.predict(X_train))
    test_mae = metrics.mean_absolute_error(y_test,clf.predict(X_test))
    test_mse = metrics.mean_squared_error(y_test, clf.predict(X_test))

    with hyper_param_list_lock:
        hyper_param_list.append((neuron_count_1, neuron_count_2, train_mae, test_mae, test_mse))

    with print_lock:
        iter += 1
        print('Iteration: ' + str(iter))
        print('Neurons 1: ' + str(neuron_count_1))
        print('Neurons 2: ' + str(neuron_count_2))
        print('\t' + 'Train Mean Absolute Error: ' + str(train_mae))
        print('\t' + 'Test Mean Absolute Error: ' + str(test_mae))
        print('\t' + 'Test Mean Squared Error: ' + str(test_mse))
        print('\n')

def threader():
    while True:
        neuron_count_1, neuron_count_2 = q.get()
        calc_mlp(neuron_count_1, neuron_count_2)
        q.task_done()

#Alle Threads starten
for i in range(12):
    t = threading.Thread(target=threader)
    t.daemon = True
    t.start()

neuron_count_1_vals = [4,8,16,32,64]
neuron_count_2_vals = [4,8,16,32,64]
for i in range(len(neuron_count_1_vals)):
    for j in range(len(neuron_count_2_vals)):
        neuron_count_1, neuron_count_2 = neuron_count_1_vals[i], neuron_count_2_vals[j]
        q.put((neuron_count_1, neuron_count_2))


#Auf alle Threads warten
q.join()

lowest_test_mse = 99999
best_hyper_param = None
for item in hyper_param_list:
    if item[4] < lowest_test_mse:
        lowest_test_mse = item[4]
        best_hyper_param = item

print(best_hyper_param)



