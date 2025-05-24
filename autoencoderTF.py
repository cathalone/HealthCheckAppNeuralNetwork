import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


# Download the dataset
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


class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(40, activation="relu"),
      layers.Dense(20, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(40, activation="relu"),
      layers.Dense(250, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(normal_train_data, normal_train_data,
          epochs=100,
          batch_size=4,
          validation_data=(test_data, test_data),
          shuffle=True)


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()


encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()



encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()


reconstructions = autoencoder.predict(anomalous_train_data)
train_loss = tf.keras.losses.mse(reconstructions, anomalous_train_data)

plt.hist(train_loss[None,:], bins=50)
plt.show()

reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mse(reconstructions, normal_train_data)

plt.hist(train_loss[None,:], bins=50)
plt.show()


threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)


def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mse(reconstructions, data)
  return tf.math.less(loss, threshold)

preds = predict(autoencoder, test_data, threshold)

for i in range(5):
    plt.plot(test_data[i], 'b')
    plt.plot(decoded_data[i], 'r')
    plt.title("prediction: " + str(np.array(preds[i])) + ", real: " + str(test_labels[i]))
    plt.fill_between(np.arange(250), decoded_data[i], test_data[i], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

c = 0
for i in range(len(test_data)):
    if test_labels[i] == preds[i]:
        c += 1

print("Accuracy: ", c/len(test_data)*100, "%")

converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)

# Конвертация модели
tflite_model = converter.convert()

# Сохранение .tflite файла
tflite_model_path = "anomaly_detector.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Модель успешно конвертирована и сохранена как {tflite_model_path}")

threshold_value = threshold
print(f"Сохраненное значение порога (threshold): {threshold_value}")
with open("threshold.txt", "w") as f:
    f.write(str(threshold_value))

