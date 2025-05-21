import struct
import matplotlib.pyplot as plt  # Импортируем matplotlib

import numpy as np


def approximate_signal_fourier(signal, num_harmonics):
    """
    Аппроксимирует сигнал с использованием K доминирующих гармоник Фурье.

    Args:
        signal (np.array): Входной одномерный сигнал.
        num_harmonics (int): Количество доминирующих гармоник для использования.

    Returns:
        np.array: Аппроксимированный сигнал.
    """
    N = len(signal)
    if num_harmonics <= 0:
        return np.zeros_like(signal)
    if num_harmonics > N // 2:  # Не может быть больше уникальных частот
        num_harmonics = N // 2

    # 1. Прямое Преобразование Фурье
    fft_coeffs = np.fft.fft(signal)

    # 2. Отбор доминирующих гармоник
    # Вычисляем амплитуды для первой половины спектра (включая DC и Найквист, если есть)
    amplitudes = np.abs(fft_coeffs[:N // 2 + 1])  # От 0 до N/2 включительно

    # Находим индексы K крупнейших амплитуд
    # argsort возвращает индексы, которые бы отсортировали массив
    # Берем последние K индексов из отсортированных по возрастанию (т.е. самые большие)
    dominant_indices_half = np.argsort(amplitudes)[-num_harmonics:]

    # Создаем отфильтрованный спектр (заполненный нулями)
    fft_coeffs_approx = np.zeros_like(fft_coeffs, dtype=complex)

    # Заполняем отфильтрованный спектр
    for k in dominant_indices_half:
        fft_coeffs_approx[k] = fft_coeffs[k]
        # Добавляем симметричную компоненту, если это не DC (k=0) и не частота Найквиста (k=N/2 для четного N)
        if k != 0 and k != N / 2:  # N/2 актуально только если N четное
            fft_coeffs_approx[N - k] = fft_coeffs[N - k]  # или np.conj(fft_coeffs[k])

    # Если N четное и частота Найквиста (N/2) была выбрана, она уже учтена.
    # Если N нечетное, то N//2 + 1 включает все уникальные частоты.

    # 3. Обратное Преобразование Фурье
    approximated_signal = np.fft.ifft(fft_coeffs_approx)

    return np.real(approximated_signal)  # Берем реальную часть



def read_ecg_data_from_java_binary(filepath):
    """
    Читает данные ЭКГ из двоичного файла, записанного Java DataOutputStream.
    (Код этой функции остается таким же, как в предыдущем ответе)
    """
    with open(filepath, 'rb') as f:
        axis_id = struct.unpack('>i', f.read(4))[0]
        utf_length = struct.unpack('>H', f.read(2))[0]
        signal_name_bytes = f.read(utf_length)
        signal_name = signal_name_bytes.decode('utf-8')
        signal_data_length = struct.unpack('>i', f.read(4))[0]
        signal_data = []
        for _ in range(signal_data_length):
            value = struct.unpack('>d', f.read(8))[0]
            signal_data.append(value)

    print(signal_data)

    return signal_data

def split_signal(signal_data, chunk_size):
    num_full_chunks = len(signal_data) // chunk_size
    return [signal_data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_full_chunks)]


def plot_ecg_signals(signal_data):
    """
    Отображает все сигналы ЭКГ из объекта ECG в одном окне.

    Args:
        ecg_data (ECG): Объект ECG, содержащий данные сигналов.
    """

    plt.figure(figsize=(12, 8))  # Задаем размер окна графика

    num_points = len(signal_data)
    time_axis = [j for j in range(num_points)]
    plt.plot(time_axis, signal_data)

    plt.title("Сигналы ЭКГ")
    plt.ylabel("Амплитуда")
    plt.legend()  # Показать легенду (имена сигналов)
    plt.grid(True)  # Добавить сетку для лучшей читаемости
    plt.tight_layout()  # Автоматически корректирует параметры subplot для плотного размещения
    plt.show()



file_path = "E:\\Downloads\\Other Downloads\\signals\\s_MAYA_6.txt"

ecg_object = read_ecg_data_from_java_binary(file_path)

for signal in split_signal(ecg_object, 500):
    plot_ecg_signals(signal)




original_signal = ecg_object

K_harmonics = int(len(ecg_object)*0.03)

approximated_signal_k2 = approximate_signal_fourier(original_signal, K_harmonics)

# K_harmonics_all = N_samples // 10  # Больше гармоник
# approximated_signal_k_more = approximate_signal_fourier(original_signal, K_harmonics_all)

# Визуализация
plt.figure(figsize=(14, 10))

time = np.linspace(0, 60, len(original_signal), endpoint=False)

plt.subplot(2, 1, 1)
plt.plot(time, original_signal)
plt.title(f"Исходный сигнал")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, approximated_signal_k2)
plt.title(f"Аппроксимированный сигнал ({K_harmonics} доминирующие гармоники)")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid(True)


plt.tight_layout()
plt.show()

N_samples = len(original_signal)
sampling_rate = int(N_samples / 4)

# Можно также посмотреть на спектры
plt.figure(figsize=(12, 6))
original_fft = np.fft.fft(original_signal)
approx_fft_k2 = np.fft.fft(approximated_signal_k2)

xf = np.fft.fftfreq(N_samples, 1 / sampling_rate)  # Ось частот

plt.plot(xf[:N_samples // 2], np.abs(original_fft[:N_samples // 2]) / N_samples * 2, label="Спектр исходного сигнала",
         alpha=0.7)
plt.plot(xf[:N_samples // 2], np.abs(approx_fft_k2[:N_samples // 2]) / N_samples * 2,
         label=f"Спектр аппроксимации ({K_harmonics} гармоники)", linestyle='--')
plt.title("Спектры Фурье")
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплитуда")
plt.legend()
plt.grid(True)
plt.show()