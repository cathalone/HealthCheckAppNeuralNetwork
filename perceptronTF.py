import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path_data = "E:\\Downloads\\Other Downloads\\signals\\amps.csv"
file_path_labels = "E:\\Projects\\GitHub\\HealthCheckAppNeuralNetwork\\collected_labels.csv"

features_df = pd.read_csv(file_path_data, header=None, sep=',')
X = features_df.values

labels_df = pd.read_csv(file_path_labels, header=None, sep=',')
if labels_df.shape[1] == 1:
    y = labels_df.iloc[:, 0].values
else:
    y = labels_df.values.reshape(-1)


print(f"Форма признаков X: {X.shape}")
print(f"Форма меток y: {y.shape}")
print(f"Примеры меток: {y[:10]}")

if X.shape[0] != y.shape[0]:
    raise ValueError(f"Количество примеров в признаках ({X.shape[0]}) не совпадает с количеством меток ({y.shape[0]})")
if X.shape[1] != 250:
    print(f"ПРЕДУПРЕЖДЕНИЕ: Ожидалось 250 признаков, получено {X.shape[1]}. Проверьте загрузку данных.")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nРазмеры выборок после разделения и масштабирования:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

input_length = 250

model = keras.Sequential(
    [
        layers.Input(shape=(input_length,)),
        layers.Dense(128, activation="relu", name="1"),
        layers.Dense(64, activation="relu", name="2"),
        layers.Dense(1, activation="sigmoid", name="3")
    ]
)

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

print("\nНачало обучения модели...")
batch_size = 4
epochs = 50

history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    verbose=1
)

print("\nОбучение завершено.")

print("\nОценка модели на тестовых данных:")
results = model.evaluate(X_test, y_test, verbose=0)

print(f"Потери на тесте: {results[0]:.4f}")
for name, value in zip(model.metrics_names[1:], results[1:]):
    print(f"{name.capitalize()}: {value:.4f}")

y_pred_proba = model.predict(X_test)

y_pred_classes = (y_pred_proba > 0.5).astype(int).flatten()

print("\nПримеры предсказаний (первые 10):")
for i in range(min(10, len(y_test))):
    print(f"real: {y_test[i]}, output: {y_pred_proba[i][0]:.4f}, predicted: {y_pred_classes[i]}")


import matplotlib.pyplot as plt

def plot_history(history_obj):
    # Потери
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_obj.history['loss'], label='Потери на обучении')
    plt.plot(history_obj.history['val_loss'], label='Потери на валидации')
    plt.title('Потери модели')
    plt.ylabel('Потери')
    plt.xlabel('Эпоха')
    plt.legend(loc='upper right')

    # Точность
    plt.subplot(1, 2, 2)
    plt.plot(history_obj.history['acс'], label='Точность на обучении') # Убедитесь, что имя метрики совпадает
    plt.plot(history_obj.history['val_точность'], label='Точность на валидации')
    plt.title('Точность модели')
    plt.ylabel('Точность')
    plt.xlabel('Эпоха')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

plot_history(history)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('my_perceptron_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("\nМодель сохранена как my_perceptron_model.tflite")