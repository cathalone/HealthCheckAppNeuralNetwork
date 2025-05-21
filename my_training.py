import numpy as np
import neunet as nn

train_data = np.loadtxt("train_data")
test_data = np.loadtxt("test_data")
train_labels = np.loadtxt("train_labels")
test_labels = np.loadtxt("test_labels")

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

n = nn.NeuralNetwork([140, 70,30,70, 140], ['identity', 'identity','identity', 'identity', 'sigmoid'], 0, 0.001, 1)

n.train(normal_train_data, normal_train_data, 20, 0.1, progress_bar=True)

nn.save_weights(n)