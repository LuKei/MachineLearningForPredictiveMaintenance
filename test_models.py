from sklearn import svm, metrics
import cmapss_data as data
from matplotlib import pyplot as plt
import pickle
import pandas as pd

#train-Daten werden zur Skalierung von valid- und test-Datenbenötigt
X_train, y_train = data.get_train_data('train_FD001.csv')
# X_test, y_test = data.get_valid_test_data('test_FD001.csv', 'test_RUL_FD001.csv')

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
# X_test = (X_test - mean) / std

max_RUL = 99999
df_results_SVR = pd.DataFrame(columns=['offset', 'mae', 'mse', 'max_RUL'])
df_results_MLP = pd.DataFrame(columns=['offset', 'mae', 'mse', 'max_RUL'])

for i in range(4, 130,4): #kürzeste Sequenz in valid-Daten ist 135 Zeilen
    X_valid, y_valid = data.get_valid_test_data('valid_FD001.csv', 'valid_RUL_FD001.csv', offset=i)
    X_valid = (X_valid - mean) / std

    clf_SVR = pickle.load(open('pickles/SVR_' + str(max_RUL) + '_max_RUL.p', 'rb'))
    valid_mae_SVR = metrics.mean_absolute_error(y_valid, clf_SVR.predict(X_valid))
    valid_mse_SVR = metrics.mean_squared_error(y_valid, clf_SVR.predict(X_valid))
    print('SVR')
    print('\toffset:  ' + str(i))
    print('\tSVR Mean Absolute Error: ' + str(valid_mae_SVR))
    print('\tSVR Mean Squared Error: ' + str(valid_mse_SVR))
    print('\n')
    df_results_SVR = df_results_SVR.append({'offset':i, 'mae':valid_mae_SVR, 'mse':valid_mse_SVR, 'max_RUL':max_RUL},
                                           ignore_index= True)

    clf_MLP = pickle.load(open('pickles/MLP_' + str(max_RUL) + '_max_RUL.p', 'rb'))
    valid_mae_MLP = metrics.mean_absolute_error(y_valid, clf_MLP.predict(X_valid))
    valid_mse_MLP = metrics.mean_squared_error(y_valid, clf_MLP.predict(X_valid))
    print('MLP')
    print('\toffset:  ' + str(i))
    print('\tMLP Mean Absolute Error: ' + str(valid_mae_MLP))
    print('\tMLP Mean Squared Error: ' + str(valid_mse_MLP))
    print('\n')
    df_results_MLP = df_results_MLP.append(
        {'offset': i, 'mae': valid_mae_MLP, 'mse': valid_mse_MLP, 'max_RUL': max_RUL},
        ignore_index=True)

df_results_SVR.to_csv(path_or_buf='results/SVR_' + str(max_RUL) + '_max_RUL.csv', sep=';', decimal=',', index=False)
df_results_MLP.to_csv(path_or_buf='results/MLP_' + str(max_RUL) + '_max_RUL.csv', sep=';', decimal=',', index=False)



