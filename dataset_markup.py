import pandas as pd
import matplotlib.pyplot as plt

def plot_ecg_signals_and_amps(signal_data, amp_data, fig_to_plot_on):
    fig_to_plot_on.clf()

    num_points_signal = len(signal_data)
    time_axis_signal = range(num_points_signal)

    num_points_amp = len(amp_data)
    time_axis_amp = range(num_points_amp)

    ax1 = fig_to_plot_on.add_subplot(2, 1, 1)
    ax1.set_title("Исходный сигнал")
    ax1.plot(time_axis_signal, signal_data)
    ax1.grid(True)

    ax2 = fig_to_plot_on.add_subplot(2, 1, 2)
    ax2.set_title("Спектр")
    ax2.plot(time_axis_amp, amp_data)
    ax2.grid(True)

    fig_to_plot_on.tight_layout()



path_amps_csv = "E:\\Downloads\\Other Downloads\\signals\\amps.csv"
path_signals_csv = "E:\\Downloads\\Other Downloads\\signals\\signals.csv"

ampsDF = pd.read_csv(path_amps_csv, sep=',', header=None)
amps_data = ampsDF.values

signalsDF = pd.read_csv(path_signals_csv, sep=',', header=None)
signals_data = signalsDF.values

print("Данные успешно загружены из CSV файлов.")


labels = []

WINDOW_GEOMETRY = "1000x700+50+50"
FIGURE_SIZE = (12, 8)

plt.ion()

fig = plt.figure(figsize=FIGURE_SIZE)
fig.canvas.manager.window.geometry(WINDOW_GEOMETRY)

for i in range(len(signals_data)):
    print(f"\nОбработка набора данных {i+1}/{len(signals_data)}")

    current_signal = signals_data[i]
    current_amp = amps_data[i]

    plot_ecg_signals_and_amps(current_signal, current_amp, fig)

    fig.canvas.draw_idle()
    plt.pause(0.1)

    label_input_value = ""
    while not label_input_value:
        user_input = input("Введите метку (или 'exit' для выхода, 'skip' для пропуска): ").strip()
        if user_input.lower() == 'exit':
            label_input_value = 'exit_loop'
            break
        elif user_input.lower() == 'skip':
            label_input_value = 'SKIPPED_BY_USER'
            print("Сигнал пропущен.")
            break
        elif user_input:
            label_input_value = user_input
            break
        else:
            print("Метка не может быть пустой. Пожалуйста, введите значение, 'skip' или 'exit'.")

    if label_input_value == 'exit_loop':
        print("Выход из цикла разметки по команде пользователя.")
        break

    labels.append(label_input_value)

plt.ioff()

if plt.fignum_exists(fig.number):
    plt.close(fig)

print("\n--- Собранные метки ---")
for idx, lbl in enumerate(labels):
    print(f"Сигнал {idx+1}: {lbl}")

labels_filename = "collected_labels.csv"
with open(labels_filename, "w") as f:
    for lbl in labels:
        f.write(f"{lbl}\n")
print(f"\nМетки также сохранены в файл: {labels_filename}")