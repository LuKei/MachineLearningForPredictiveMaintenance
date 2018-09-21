from sklearn import neural_network, metrics
import cmapss_data as data
import threading
from queue import Queue
import pickle
import multiprocessing

max_RUL = 99999
X_train, y_train = data.get_train_data('train_FD001.csv', maximum_RUL=max_RUL)
X_valid, y_valid = data.get_valid_test_data('valid_FD001.csv', 'valid_RUL_FD001.csv')

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std

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
    train_mse = metrics.mean_squared_error(y_train, clf.predict(X_train))
    valid_mae = metrics.mean_absolute_error(y_valid,clf.predict(X_valid))
    valid_mse = metrics.mean_squared_error(y_valid, clf.predict(X_valid))

    with hyper_param_list_lock:
        hyper_param_list.append((neuron_count_1, neuron_count_2, train_mae, train_mse, valid_mae, valid_mse))

    with print_lock:
        iter += 1
        print('Iteration: ' + str(iter))
        print('Neurons 1: ' + str(neuron_count_1))
        print('Neurons 2: ' + str(neuron_count_2))
        print('max RUL: ' + str(max_RUL))
        print('\t' + 'Train Mean Absolute Error: ' + str(train_mae))
        print('\t' + 'Train Mean Squared Error: ' + str(train_mse))
        print('\t' + 'Test Mean Absolute Error: ' + str(valid_mae))
        print('\t' + 'Test Mean Squared Error: ' + str(valid_mse))
        print('\n')

def threader():
    while True:
        neuron_count_1, neuron_count_2 = q.get()
        calc_mlp(neuron_count_1, neuron_count_2)
        q.task_done()

#Alle Threads starten
for i in range(multiprocessing.cpu_count()):
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
    if item[5] < lowest_test_mse:
        lowest_test_mse = item[5]
        best_hyper_param = item

print(best_hyper_param)


#Save the best model as pickle
clf = neural_network.MLPRegressor(hidden_layer_sizes=(best_hyper_param[0], best_hyper_param[1]), max_iter=500)
clf.fit(X_train, y_train)
pickle.dump(clf, open('pickles/MLP_' + str(max_RUL) + '_max_RUL.p', 'wb'))



