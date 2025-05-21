import neunet as nn
import numpy as np
import matplotlib.pyplot as plt

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


# n = nn.NeuralNetwork([140, 100, 70, 50, 70, 100, 140], ['identity', 'identity', 'identity','identity', 'identity', 'identity', 'sigmoid'], 0, 0.001, 1)

n = nn.NeuralNetwork([140, 70,30,70, 140], ['identity', 'identity','identity', 'identity', 'sigmoid'], 0, 0.001, 1)

nn.load_weights(n)


decoded_normal_data = []
for i in normal_test_data:
    decoded_normal_data.append(n.check(i))

# for i in range(10):
#     plt.plot(normal_test_data[i], 'b')
#     plt.plot(decoded_normal_data[i], 'r')
#     plt.title("Нормальная ЭКГ")
#     plt.fill_between(np.arange(140), decoded_normal_data[i], normal_test_data[i], color='lightcoral')
#     plt.legend(labels=["Input", "Reconstruction", "Error"])
#     plt.show()

decoded_anomalous_data = []
for i in anomalous_test_data:
    decoded_anomalous_data.append(n.check(i))

# for i in range(10):
#     plt.plot(anomalous_test_data[i], 'b')
#     plt.plot(decoded_anomalous_data[i], 'r')
#     plt.title("Аномальная ЭКГ")
#     plt.fill_between(np.arange(140), decoded_anomalous_data[i], anomalous_test_data[i], color='lightcoral')
#     plt.legend(labels=["Input", "Reconstruction", "Error"])
#     plt.show()

decoded_normal_train_data = []
for i in normal_train_data:
    decoded_normal_train_data.append(n.check(i))

loss = []
for i in range(len(decoded_normal_train_data)):
    loss.append(nn.loss_func(decoded_normal_train_data[i], normal_train_data[i]))

threshold = np.mean(loss) + np.std(loss)
print("Threshold: ", threshold)

decoded_data = []
for i in test_data:
    decoded_data.append(n.check(i))

predicted = []
for i in range(len(test_data)):
    if nn.loss_func(decoded_data[i], test_data[i]) > threshold:
        predicted.append(False)
    else:
        predicted.append(True)

c = 0
for i in range(len(test_data)):
    if test_labels[i] == predicted[i]:
        c += 1

for i in range(10):
    plt.plot(test_data[i], 'b')
    plt.plot(decoded_data[i], 'r')
    plt.title("prediction: " + str(predicted[i]) + ", real: " + str(test_labels[i]))
    plt.fill_between(np.arange(140), decoded_data[i], test_data[i], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

print("Accuracy: ", c/len(test_data)*100, "%")