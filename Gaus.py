from sklearn import svm, cross_validation
from sklearn.naive_bayes import GaussianNB

import debs_data as data
import numpy as np


sensor_data = data.get_sensor_data_array('molding_machine_5000dp.csv', machine_id=59)
anomaly_labels = data.get_anomaly_array('molding_machine_5000dp.anomalies.nt', machine_id=59,
                                          sensor_data_count=len(sensor_data))


sensor_data_only_anomalies = data.get_sensor_data_array('molding_machine_5000dp.csv', machine_id=59,
                                                        only_anomalies=True,
                                                        anomaly_filename='molding_machine_5000dp.anomalies.nt')

for i in range(20, 81, 20):

    print('\t' + 'Limit: ' + str(i))
    limit = int(len(sensor_data) * i / 100)

    X_train = sensor_data[0:limit]
    X_test = sensor_data[limit:]
    y_train = anomaly_labels[0:limit]
    y_test = anomaly_labels[limit:]

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print('\t' + 'Accuracy: ' + str(accuracy))

    prediction = clf.predict(sensor_data_only_anomalies)
    print('\t' + 'Prediction: ' + str(prediction))
    accuracy_predic = np.sum(prediction) / len(prediction)
    print('\t' + 'Prediction accuracy: ' + str(accuracy_predic) +'\n')





