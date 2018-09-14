from sklearn import svm, model_selection, metrics
import debs_data as data
import numpy as np
import math

sensor_data = data.get_sensor_data_array('molding_machine_5000dp.csv', machine_id=59,
                                         anomalies='include', anomaly_filename='molding_machine_5000dp.anomalies.nt')

anomaly_labels = data.get_anomaly_array('molding_machine_5000dp.anomalies.nt', machine_id=59,
                                          sensor_data_count=len(sensor_data))

X_train, X_test, y_train, y_test = model_selection.train_test_split(sensor_data, anomaly_labels, train_size=0.8, test_size=0.2)


nu = np.linspace(start=0.0001, stop=0.5, num=20)
gamma = np.linspace(start=0.00001, stop=0.8, num=20)
opt_diff = 1.0
opt_nu = None
opt_gamma = None

k = 0
#for i in range(len(nu)):
 #   for j in range(len(gamma)):

k += 1
print('Iteration: ' + str(k))
#print('Nu: ' + str(nu[i]))
#print('Gamma: ' + str(gamma[j]))


clf = svm.OneClassSVM(kernel="rbf", nu=0.47368947368421055, gamma=1e-05)
clf.fit(X_train)

preds = clf.predict(X_train)
targs = y_train

p = 1 - float(sum(preds == 1.0)) / len(preds)
# diff = math.fabs(p - nu[i])
# if diff < opt_diff:
#     opt_diff = diff
#     opt_nu = nu[i]
#     opt_gamma = gamma[j]

preds = clf.predict(X_test)
targs = y_test

print("accuracy: ", metrics.accuracy_score(targs, preds))
print("precision: ", metrics.precision_score(targs, preds))
print("recall: ", metrics.recall_score(targs, preds))
print("f1: ", metrics.f1_score(targs, preds))
print("area under curve (auc): ", metrics.roc_auc_score(targs, preds))
print('\n')

print('Opt nu: ' + str(opt_nu))
print('Opt gamma: ' + str(opt_gamma))






