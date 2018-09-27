from sklearn import metrics
import cmapss_data as data
import pickle
import pandas as pd
from matplotlib import pyplot as plt

model = 'SVR'
max_RULs = [80,90,100,99999]

plt.figure(figsize=(10, 10))

i = 1
for max_RUL in max_RULs:
    #train-Daten werden zur Skalierung von valid- und test-Datenbenötigt
    X_train, y_train = data.get_train_data('train_FD001.csv', maximum_RUL=max_RUL)
    X_test, y_test = data.get_valid_test_data('test_FD001.csv', 'test_RUL_FD001.csv', maximum_RUL=max_RUL)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_test = (X_test - mean) / std

    clf = pickle.load(open('pickles/' + model + '_' + str(max_RUL) + '_max_RUL.p', 'rb'))
    test_mae = metrics.mean_absolute_error(y_test, clf.predict(X_test))
    test_mse = metrics.mean_squared_error(y_test, clf.predict(X_test))
    print('RUL: ' + str(max_RUL))
    print(model + ' Test Mean Absolute Error: ' + str(test_mae))
    print(model + ' Test Mean Squared Error: ' + str(test_mse))

    y_predicted = clf.predict(X_test)


    plt.subplot(2, 2, i)
    plt.plot([0,300], [0,300])
    plt.plot(y_predicted, y_test, 'b.')
    plt.xlabel('vorhergesagte RUL', fontsize=16)
    plt.ylabel('tatsächliche RUL',fontsize=16)
    plt.title('Maximale RUL: ' + (str(max_RUL) if max_RUL <= 110 else 'ohne'), fontsize=16)
    plt.gca().tick_params(labelsize=16)

    i += 1

plt.subplots_adjust(top=0.97, bottom=0.07, left=0.12, right=0.9, hspace=0.3,wspace=0.3)
plt.show()


# for max_RUL in max_RULs:
#     #train-Daten werden zur Skalierung von valid- und test-Datenbenötigt
#     X_train, y_train = data.get_train_data('train_FD001.csv', maximum_RUL=max_RUL)
#     X_test, y_test = data.get_valid_test_data('test_FD001.csv', 'test_RUL_FD001.csv', maximum_RUL=max_RUL,
#                                               stop_at_RUL=40)
#
#     mean = X_train.mean(axis=0)
#     std = X_train.std(axis=0)
#     X_test = (X_test - mean) / std
#
#     clf = pickle.load(open('pickles/' + model + '_' + str(max_RUL) + '_max_RUL.p', 'rb'))
#     test_mae = metrics.mean_absolute_error(y_test, clf.predict(X_test))
#     test_mse = metrics.mean_squared_error(y_test, clf.predict(X_test))
#     print('RUL: ' + str(max_RUL))
#     print(model + ' Test Mean Absolute Error: ' + str(test_mae))
#     print(model + ' Test Mean Squared Error: ' + str(test_mse))




