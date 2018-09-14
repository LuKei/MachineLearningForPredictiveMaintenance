import tensorflow as tf
import threading
from queue import Queue
import cmapss_data as data
import numpy as np


X_train, X_test, y_train, y_test = data.get_sensor_data('train_FD001.csv','test_FD001.csv', 'RUL_FD001.csv')
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

hyper_param_list = []
iter = 0

q = Queue()

def calc_nn(neuron_count_1, neuron_count_2):
    global iter, hyper_param_list
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(neuron_count_1, activation=tf.nn.relu, input_shape=(6,)))
    model.add(tf.keras.layers.Dense(neuron_count_2, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    model.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=0)
    loss_test, mae_test = model.evaluate(X_test, y_test, verbose=0)
    loss_train, mae_train = model.evaluate(X_test, y_test, verbose=0)
    prediction_RUL = model.predict(X_test[0:1])
    real_RUL = [y_test[0]]

    hyper_param_list.append((neuron_count_1, neuron_count_2, mae_train, mae_test))

    iter += 1
    print('Iteration: ' + str(iter))
    print('Neurons 1: ' + str(neuron_count_1))
    print('Neurons 2: ' + str(neuron_count_2))
    print('\t' + 'Train Mean Absolute Error: ' + str(mae_train))
    print('\t' + 'Test Mean Absolute Error: ' + str(mae_test))
    print('\t' + 'Prediction RUL: ' + str(prediction_RUL) + '\n')
    print('\t' + 'Real RUL: ' + str(real_RUL) + '\n')
    print('\n')




neuron_count_1_vals = np.linspace(start=2**3, stop=2**8, num=4, dtype=int)
neuron_count_2_vals = np.linspace(start=2**3, stop=2**8, num=4, dtype=int)
for i in range(len(neuron_count_1_vals)):
    for j in range(len(neuron_count_2_vals)):
        neuron_count_1, neuron_count_2 = neuron_count_1_vals[i], neuron_count_2_vals[j]
        calc_nn(neuron_count_1, neuron_count_2)



best_test_acc = -1
best_hyper_param = None
for item in hyper_param_list:
    if item[3] > best_test_acc:
        best_test_acc = item[3]
        best_hyper_param = item

print(best_hyper_param)


