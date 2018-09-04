import tensorflow as tf
import numpy as np
import debs_data as data

#tf.enable_eager_execution()

#Training data
train_sensor_data = data.get_sensor_data_array('molding_machine_5000dp.csv')
train_anomaly_labels = data.get_anomaly_array('molding_machine_5000dp.anomalies.nt', len(train_sensor_data))

#Testing data
test_sensor_data = data.get_sensor_data_array('molding_machine_308dp.csv')
test_anomaly_labels = data.get_anomaly_array('molding_machine_308dp.anomalies.nt', len(test_sensor_data))

#Build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu, input_shape=(117,)))
#model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#Train the model
model.fit(train_sensor_data, train_anomaly_labels, epochs=5)

#Test accuracy
loss, accuracy = model.evaluate(test_sensor_data, test_anomaly_labels)
print('Accuracy', accuracy)

#Predict
scores = model.predict(test_sensor_data[0:1]) #test for firts sensor data set
print(np.argmax(scores))

