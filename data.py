import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

ampsDF = pd.read_csv("E:\\Downloads\\Other Downloads\\signals\\amps.csv")
amps_data = ampsDF.values
data = amps_data[:, 0:-1]

labelsDF = pd.read_csv("E:\\Projects\\GitHub\\HealthCheckAppNeuralNetwork\\collected_labels.csv")
labels = labelsDF.iloc[:,0].values

print(labels)




train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)


train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

np.savetxt("train_data", train_data)
np.savetxt("test_data", test_data)
np.savetxt("train_labels", train_labels)
np.savetxt("test_labels", test_labels)
