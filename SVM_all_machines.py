from sklearn import svm, cross_validation
import debs_data as data
import numpy as np


#Training data
sensor_data_dic = {}
anomaly_labels_dic = {}
for i in range(0, 9):
    sensor_data = data.get_sensor_data_array('10molding_machine_5000dp.csv', machine_id=i)
    anomaly_labels = data.get_anomaly_array('10molding_machine_5000dp.anomalies.nt', machine_id=i,
                                              sensor_data_count=len(sensor_data))
    sensor_data_dic[i] = sensor_data
    anomaly_labels_dic[i] = anomaly_labels


sensor_data_only_anomalies = data.get_sensor_data_array('10molding_machine_5000dp.csv', machine_id=9,
                                                        only_anomalies=True,
                                                        anomaly_filename='10molding_machine_5000dp.anomalies.nt')



X_train = np.concatenate((sensor_data_dic[0], sensor_data_dic[1], sensor_data_dic[2], sensor_data_dic[3],
                          sensor_data_dic[4], sensor_data_dic[5], sensor_data_dic[6], sensor_data_dic[7],
                          sensor_data_dic[8]))
y_train = np.concatenate((anomaly_labels_dic[0], anomaly_labels_dic[1], anomaly_labels_dic[2],anomaly_labels_dic[3],
                          anomaly_labels_dic[4], anomaly_labels_dic[5], anomaly_labels_dic[6],anomaly_labels_dic[7],
                         anomaly_labels_dic[8]))


clf = svm.SVC()
clf.fit(X_train, y_train)

prediction = clf.predict(sensor_data_only_anomalies)
print('\t' + 'Prediction: ' + str(prediction))
accuracy_predic = np.sum(prediction) / len(prediction)
print('\t' + 'Prediction accuracy: ' + str(accuracy_predic))





