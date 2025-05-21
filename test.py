# 1
import cv2
import os
import requests  # Для скачивания каскада, если его нет
import numpy as np
import matplotlib.pyplot as plt  # Для отображения в средах типа Jupyter

# --- Константы и пути ---
# URL каскада для кошачьих морд (расширенная версия)
# Вы можете поискать каскады для собак, если найдете готовые и качественные.
# Например, на GitHub иногда встречаются пользовательские.
# Для демонстрации возьмем кошачий, т.к. он есть в официальном репозитории OpenCV.
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalcatface_extended.xml"
CASCADE_FILENAME = "haarcascade_frontalcatface_extended.xml"


# --- Вспомогательные функции ---

def download_cascade(url, filename):
    """Скачивает файл, если он отсутствует."""
    if not os.path.exists(filename):
        print(f"Скачивание {filename}...")
        try:
            r = requests.get(url, allow_redirects=True)
            r.raise_for_status()  # Проверка на ошибки HTTP
            with open(filename, 'wb') as f:
                f.write(r.content)
            print(f"{filename} успешно скачан.")
        except requests.exceptions.RequestException as e:
            print(f"Ошибка скачивания {filename}: {e}")
            return False
    return True


def display_image(image, title="Image"):
    """Отображает изображение с помощью Matplotlib (удобно для Jupyter)."""
    # OpenCV читает в BGR, Matplotlib ожидает RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()


# --- Основная логика ---

class PetFaceDetector:
    def __init__(self, cascade_path):
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Файл каскада не найден: {cascade_path}")

        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise IOError(f"Не удалось загрузить каскад из {cascade_path}")
        print(f"Каскад {cascade_path} успешно загружен.")

    def detect_faces(self, image_path,
                     scale_factor=1.1,
                     min_neighbors=5,
                     min_size=(30, 30)):
        """
        Распознает мордочки на изображении.

        Args:
            image_path (str): Путь к изображению.
            scale_factor (float): Параметр detectMultiScale. Определяет, насколько
                                  уменьшается размер изображения на каждом шаге масштабирования.
            min_neighbors (int): Параметр detectMultiScale. Определяет, сколько "соседей"
                                 (перекрывающихся обнаружений) должно быть у каждого кандидата
                                 в прямоугольники, чтобы он считался истинным обнаружением.
            min_size (tuple): Минимальный размер обнаруживаемого объекта.

        Returns:
            tuple: (image_with_boxes, list_of_boxes)
                   image_with_boxes: Изображение с нарисованными рамками (numpy array).
                   list_of_boxes: Список кортежей (x, y, w, h) для каждой обнаруженной мордочки.
        """
        if not os.path.exists(image_path):
            print(f"Файл изображения не найден: {image_path}")
            return None, []

        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось прочитать изображение: {image_path}")
            return None, []

        # Для каскадов Хаара обычно лучше работать с grayscale изображением
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Обнаружение морд
        # Параметры detectMultiScale можно и нужно подбирать для конкретного набора данных
        # и типа объектов для улучшения качества детекции.
        faces = self.detector.detectMultiScale(
            gray_image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE  # Флаг по умолчанию
        )

        print(f"Найдено {len(faces)} потенциальных морд(ы).")

        # Рисуем прямоугольники вокруг найденных морд
        image_with_boxes = image.copy()  # Работаем с копией, чтобы не изменять оригинал
        detected_boxes = []
        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected_boxes.append((x, y, w, h))
            # Можно добавить текст
            # cv2.putText(image_with_boxes, 'Pet Face', (x, y-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        return image_with_boxes, detected_boxes


# --- Пример использования ---
if __name__ == "__main__":
    # 1. Скачиваем каскад, если его нет
    if not download_cascade(CASCADE_URL, CASCADE_FILENAME):
        print("Не удалось получить файл каскада. Выполнение прервано.")
        exit()

    # 2. Инициализируем детектор
    try:
        detector = PetFaceDetector(CASCADE_FILENAME)
    except (FileNotFoundError, IOError) as e:
        print(f"Ошибка инициализации детектора: {e}")
        exit()

    # 3. Подготовьте тестовые изображения
    #    Поместите ваши изображения в папку 'test_images' или укажите прямые пути
    #    Для примера, создадим фейковое изображение, если у вас нет готовых.
    #    В реальном сценарии вы бы использовали изображения из датасета "Dog Face Recognition"

    # Создаем папку для тестовых изображений, если ее нет
    test_image_dir = "test_images_pets"
    os.makedirs(test_image_dir, exist_ok=True)

    # Пример: скачаем тестовое изображение кота
    test_cat_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    test_cat_image_path = os.path.join(test_image_dir, "test_cat.jpg")

    if not os.path.exists(test_cat_image_path):
        print(f"Скачивание тестового изображения кота в {test_cat_image_path}...")
        try:
            r_img = requests.get(test_cat_image_url)
            r_img.raise_for_status()
            with open(test_cat_image_path, 'wb') as f:
                f.write(r_img.content)
            print("Тестовое изображение кота скачано.")
        except requests.exceptions.RequestException as e:
            print(f"Не удалось скачать тестовое изображение: {e}")
            # Если скачать не удалось, можно создать простое синтетическое изображение
            # но на нем каскад вряд ли что-то найдет.
            # For now, we'll just proceed and it will likely fail on detection if image is missing.

    # Пример: скачаем тестовое изображение собаки
    # ВАЖНО: используемый каскад haarcascade_frontalcatface_extended.xml обучен на кошках.
    # На собаках он может работать плохо или не работать вовсе.
    # Для собак нужен каскад, обученный на собаках!
    test_dog_image_url = "https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_3x2.jpg"
    test_dog_image_path = os.path.join(test_image_dir, "test_dog.jpg")

    if not os.path.exists(test_dog_image_path):
        print(f"Скачивание тестового изображения собаки в {test_dog_image_path}...")
        try:
            r_img_dog = requests.get(test_dog_image_url)
            r_img_dog.raise_for_status()
            with open(test_dog_image_path, 'wb') as f:
                f.write(r_img_dog.content)
            print("Тестовое изображение собаки скачано.")
        except requests.exceptions.RequestException as e:
            print(f"Не удалось скачать тестовое изображение собаки: {e}")

    sample_images = []
    if os.path.exists(test_cat_image_path):
        sample_images.append(test_cat_image_path)
    if os.path.exists(test_dog_image_path):
        sample_images.append(test_dog_image_path)

    if not sample_images:
        print(f"Пожалуйста, поместите изображения в папку '{test_image_dir}' или укажите пути.")
        print("Или убедитесь, что тестовые изображения были успешно скачаны.")
    else:
        for img_path in sample_images:
            print(f"\n--- Обработка изображения: {img_path} ---")
            # Можно поиграть с параметрами scale_factor, min_neighbors, min_size
            # для разных изображений
            image_with_detections, boxes = detector.detect_faces(
                img_path,
                scale_factor=1.05,  # Более мелкий шаг может дать больше детекций, но медленнее
                min_neighbors=3,  # Меньшее значение -> больше детекций (и ложных срабатываний)
                min_size=(50, 50)  # Игнорировать очень маленькие объекты
            )

            if image_with_detections is not None:
                # Отображение с помощью Matplotlib (удобно для Jupyter/Colab)
                display_image(image_with_detections, title=f"Обнаруженные мордочки на {os.path.basename(img_path)}")

                # Если вы не в Jupyter, можете использовать cv2.imshow:
                # cv2.imshow(f"Detections on {os.path.basename(img_path)}", image_with_detections)
                # cv2.waitKey(0) # Ждать нажатия клавиши
                # cv2.destroyWindow(f"Detections on {os.path.basename(img_path)}")

                if boxes:
                    print("Координаты рамок (x, y, ширина, высота):")
                    for box in boxes:
                        print(box)
                else:
                    print("Мордочки не найдены на этом изображении с текущими параметрами.")

        # cv2.destroyAllWindows() # Закрыть все окна OpenCV, если использовался cv2.imshow

    print("\n--- Важные замечания ---")
    print("1. Использовался каскад для КОШАЧЬИХ морд. Для собак он может работать плохо.")
    print("2. Для качественного распознавания СОБАЧЬИХ морд необходимо:")
    print("   а) Найти готовый качественный каскад Хаара для собак (сложно).")
    print("   б) Обучить собственный каскад Хаара на датасете 'Dog Face Recognition'.")
    print("      Это включает сбор позитивных (морды собак) и негативных (без морд) выборок,")
    print("      использование утилит opencv_createsamples и opencv_traincascade.")
    print("3. Каскады Хаара менее точны, чем современные нейросетевые подходы (как в статье на Habr),")
    print("   но не требуют TensorFlow/PyTorch и могут быть быстрее на CPU для простых задач.")
    print("4. Параметры `scale_factor`, `min_neighbors`, `min_size` в `detect_faces` сильно влияют")
    print("   на результат и требуют подбора.")


# 2

import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import requests  # Для скачивания изображений


# --- Вспомогательные функции ---

def display_image_with_boxes(image_path_or_pil, boxes, labels=None, title="Detections"):
    """Отображает изображение с рамками."""
    if isinstance(image_path_or_pil, str):
        img_pil = Image.open(image_path_or_pil).convert("RGB")
    else:  # Предполагаем, что это PIL Image
        img_pil = image_path_or_pil.convert("RGB")

    img_cv = np.array(img_pil)
    # img_cv = img_cv[:, :, ::-1].copy() # RGB to BGR, но matplotlib ожидает RGB

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        if labels and i < len(labels):
            label_text = labels[i]
            cv2.putText(img_cv, label_text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_cv)  # Matplotlib ожидает RGB
    plt.title(title)
    plt.axis('off')
    plt.show()


# --- Класс детектора ---

class PetFacePyTorchDetector:
    def __init__(self, model_path=None, num_classes_custom=None, device=None):
        """
        model_path: Путь к сохраненной кастомной модели. Если None, используется предобученная COCO.
        num_classes_custom: Количество классов для кастомной модели (включая фон).
        device: 'cuda' или 'cpu'
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if model_path and num_classes_custom:
            print(f"Loading custom model from {model_path} with {num_classes_custom} classes.")
            # Загрузка кастомной модели (пример)
            # Предполагается, что архитектура соответствует FasterRCNN с ResNet50 FPN
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, progress=False)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                                            num_classes_custom)
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Custom model loaded successfully.")
            except Exception as e:
                print(f"Error loading custom model: {e}")
                print("Falling back to COCO pretrained model for 'dog'/'cat' detection.")
                self._load_coco_model()
            # Вам нужно будет определить свои метки классов для кастомной модели
            self.custom_class_names = ["background", "pet_face"]  # Пример
        else:
            print("Loading COCO pretrained model for 'dog'/'cat' detection.")
            self._load_coco_model()

        self.model.eval()  # Переводим модель в режим оценки


self.model.to(self.device)


def _load_coco_model(self):
    """Загружает предобученную на COCO модель."""
    self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    self.using_coco_model = True
    # Классы COCO (Faster R-CNN из torchvision обучен на них)
    self.COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    self.pet_class_indices_coco = [self.COCO_INSTANCE_CATEGORY_NAMES.index('cat'),
                                   self.COCO_INSTANCE_CATEGORY_NAMES.index('dog')]


def detect_faces(self, image_path_or_pil, confidence_threshold=0.5):
    """
    Распознает объекты на изображении.
    Если используется COCO модель, ищет 'cat' и 'dog'.
    Если кастомная модель, ищет 'pet_face'.
    """
    if isinstance(image_path_or_pil, str):
        try:
            img_pil = Image.open(image_path_or_pil).convert("RGB")
        except FileNotFoundError:
            print(f"Файл не найден: {image_path_or_pil}")
            return [], []
        except Exception as e:
            print(f"Ошибка загрузки изображения {image_path_or_pil}: {e}")
            return [], []
    elif isinstance(image_path_or_pil, Image.Image):
        img_pil = image_path_or_pil.convert("RGB")
    else:
        print("Неверный тип входных данных для изображения.")
        return [], []

    # Преобразование изображения в тензор
    img_tensor = F.to_tensor(img_pil)
    img_tensor = img_tensor.to(self.device)

    with torch.no_grad():  # Отключаем расчет градиентов для инференса
        predictions = self.model([img_tensor])

    pred = predictions[0]

    pred_boxes = []
    pred_labels_text = []
    # pred_scores = [] # Если нужны числовые значения уверенности

    for i in range(len(pred['scores'])):
        score = pred['scores'][i].item()
        if score > confidence_threshold:
            label_idx = pred['labels'][i].item()
            box = pred['boxes'][i].cpu().numpy()  # [xmin, ymin, xmax, ymax]

            if hasattr(self, 'using_coco_model') and self.using_coco_model:
                if label_idx in self.pet_class_indices_coco:
                    pred_boxes.append(box)
                    pred_labels_text.append(self.COCO_INSTANCE_CATEGORY_NAMES[label_idx])
                    # pred_scores.append(score)
            else:  # Кастомная модель
                # Предполагаем, что класс "pet_face" имеет индекс 1 (0 - фон)
                if label_idx == 1:  # Или другой индекс вашего класса мордочки
                    pred_boxes.append(box)
                    pred_labels_text.append(self.custom_class_names[label_idx])
                    # pred_scores.append(score)

    if hasattr(self, 'using_coco_model') and self.using_coco_model:
        print(f"Найдено {len(pred_boxes)} питомцев (собак/кошек из COCO) с уверенностью > {confidence_threshold}")
    else:
        print(f"Найдено {len(pred_boxes)} мордочек (из кастомной модели) с уверенностью > {confidence_threshold}")

    return pred_boxes, pred_labels_text


# --- Функции для скачивания и подготовки изображений ---
def download_image(url, path):
    if not os.path.exists(path):
        print(f"Скачивание {os.path.basename(path)}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Скачано.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Не удалось скачать {os.path.basename(path)}: {e}")
            return False
    return True


# --- Пример использования ---
if __name__ == "__main__":
    # УКАЖИТЕ ПУТЬ К ВАШЕЙ ОБУЧЕННОЙ МОДЕЛИ, ЕСЛИ ЕСТЬ
    # Иначе будет использована COCO модель
    # custom_model_path = "path/to/your/trained_pet_face_detector.pth"
    custom_model_path = None  # Для демонстрации с COCO
    num_classes_for_custom_model = 2  # (фон, мордочка_питомца)

    if custom_model_path and os.path.exists(custom_model_path):
        detector = PetFacePyTorchDetector(model_path=custom_model_path, num_classes_custom=num_classes_for_custom_model)
    else:
        if custom_model_path:  # Если путь указан, но файла нет
            print(f"Файл кастомной модели не найден: {custom_model_path}. Используется COCO модель.")
        detector = PetFacePyTorchDetector()

    # Подготовьте тестовые изображения
    test_image_dir = "test_images_pets_pytorch"
    os.makedirs(test_image_dir, exist_ok=True)

    image_urls = {
        "cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "dog": "https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_3x2.jpg",
        "dogs_group": "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
        "cat_face_close": "https://hips.hearstapps.com/hmg-prod/images/cute-cat-photos-1590542211.jpg"
        # Для проверки на мордочках
    }

    sample_image_paths = []
    for name, url in image_urls.items():
        path = os.path.join(test_image_dir, f"test_{name}.jpg")
        if download_image(url, path):
            sample_image_paths.append(path)

    if not sample_image_paths:
        print(f"Не удалось скачать тестовые изображения или папка '{test_image_dir}' пуста.")
    else:
        for img_path in sample_image_paths:
            print(f"\n--- Обработка изображения: {img_path} ---")

            # Детекция
            boxes, labels = detector.detect_faces(img_path, confidence_threshold=0.6)

            if boxes:
                display_image_with_boxes(img_path, boxes, labels, title=f"Обнаружения на {os.path.basename(img_path)}")
            else:
                print("Объекты не найдены или уверенность слишком низкая.")
                # Показать оригинальное изображение, если ничего не найдено
                try:
                    img_pil = Image.open(img_path).convert("RGB")
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img_pil)
                    plt.title(f"Нет детекций на {os.path.basename(img_path)}")
                    plt.axis('off')
                    plt.show()
                except Exception as e:
                    print(f"Не удалось отобразить изображение {img_path}: {e}")

    print("\n--- Важные замечания по этому примеру ---")
    print(
        "1. Если не указан `custom_model_path`, используется Faster R-CNN (ResNet50 FPN), предобученная на COCO. Она детектирует общие классы 'cat' и 'dog'.")
    print("2. Эта COCO-модель детектирует всего животного, а не только мордочку.")
    print("3. Для детекции именно МОРДОЧЕК питомцев, необходимо:")
    print(
        "   а) Подготовить датасет 'Dog Face Recognition' (и аналогичный для кошек, если нужно) с аннотациями bounding box'ов именно морд.")
    print(
        "   б) Взять предобученную модель детекции (как Faster R-CNN, SSD, YOLO) и дообучить (fine-tune) ее на вашем датасете морд.")
    print("      Это включает замену 'головы' классификатора модели для предсказания вашего класса ('pet_face').")
    print(
        "   в) Написать полный цикл обучения, включая класс Dataset, DataLoader, функцию потерь, оптимизатор (не показано в этом скрипте).")
    print("4. Этот код демонстрирует инференс с помощью PyTorch и torchvision.")
    print(
        "5. Полное обучение модели на PyTorch - это отдельная большая задача, требующая подготовки данных и написания тренировочного цикла.")
    print(
        "6. Если вы обучите свою модель, укажите путь к ней в `custom_model_path` и правильное `num_classes_for_custom_model`.")


# 2.1

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 2 # 1 класс "мордочка" + 1 класс "фон"
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 3

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator  # Для MobileNetV3-FPN
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd  # Для чтения CSV аннотаций
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # Для рисования рамок


# --- Вспомогательные функции ---

def display_image_with_boxes_pil(pil_image, boxes, labels=None, title="Detections", scores=None, score_threshold=0.5):
    """Отображает PIL изображение с рамками."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(pil_image)
    ax.set_title(title)
    ax.axis('off')

    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < score_threshold:
            continue

        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        if labels and i < len(labels):
            label_text = labels[i]
            if scores is not None:
                label_text += f": {scores[i]:.2f}"
            ax.text(xmin, ymin - 5, label_text, color='lime', fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5, pad=0.2, edgecolor='none'))
    plt.show()


# --- Класс Dataset ---
class DogFaceDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        """
        Args:
            root_dir (string): Путь к папке train (например, 'dog-face-recognition/train').
            annotation_file (string): Путь к CSV файлу с аннотациями
                                      (например, 'dog-face-recognition/train_annotations.csv').
            transforms (callable, optional): Трансформации для изображений и аннотаций.
        """
        self.root_dir = root_dir
        self.transforms = transforms

        # Загрузка аннотаций
        # ПРЕДПОЛОЖЕНИЕ: CSV файл с колонками 'filepath', 'xmin', 'ymin', 'xmax', 'ymax'
        # filepath - относительный путь от root_dir, например, '000001/0001.jpg'
        try:
            self.annotations_df = pd.read_csv(annotation_file)
        except FileNotFoundError:
            print(f"Файл аннотаций не найден: {annotation_file}")
            print("Пожалуйста, убедитесь, что файл существует и путь указан верно.")
            print("Без аннотаций обучение невозможно.")
            raise
        except Exception as e:
            print(f"Ошибка при чтении файла аннотаций {annotation_file}: {e}")
            raise

        # Группируем аннотации по файлам, так как одно изображение может иметь несколько морд
        self.image_infos = []
        for img_path, group in self.annotations_df.groupby('filepath'):
            full_img_path = os.path.join(self.root_dir, img_path)
            if not os.path.exists(full_img_path):
                print(f"Предупреждение: Файл изображения {full_img_path} из аннотаций не найден. Пропускается.")
                continue

            boxes = group[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
            # Метка класса "мордочка" будет 1 (0 - фон)
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

            self.image_infos.append({
                'image_path': full_img_path,
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': labels
            })

        if not self.image_infos:
            print("Не найдено валидных изображений/аннотаций. Проверьте пути и файл аннотаций.")

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        img_info = self.image_infos[idx]

        try:
            image = Image.open(img_info['image_path']).convert("RGB")
        except FileNotFoundError:
            print(f"Ошибка: Файл {img_info['image_path']} не найден во время __getitem__.")
            # Это не должно происходить, если проверка в __init__ сработала
            # Можно вернуть None или обработать иначе
            return None, None  # Или поднять исключение
        except Exception as e:
            print(f"Ошибка загрузки изображения {img_info['image_path']}: {e}")
            return None, None

        target = {
            'boxes': img_info['boxes'],
            'labels': img_info['labels'],
            'image_id': torch.tensor([idx])  # Уникальный ID для изображения
        }

        # torchvision модели детекции ожидают, что target['area'] и target['iscrowd']
        # будут присутствовать, если вы используете стандартные evaluation утилиты.
        # Для простоты обучения их можно опустить, но лучше добавить.
        num_objs = len(img_info['labels'])
        areas = []
        for i in range(num_objs):
            box = img_info['boxes'][i]
            areas.append((box[2] - box[0]) * (box[3] - box[1]))
        target['area'] = torch.as_tensor(areas, dtype=torch.float32)
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)  # Предполагаем, что нет "толп" объектов

        if self.transforms:
            # Важно: некоторые трансформации (например, Resize) должны применяться
            # и к изображению, и к bounding box'ам.
            # Простой ToTensor + Normalize:
            # image = TF.to_tensor(image)
            # image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Для более сложных (как в torchvision detection tutorial) нужно передавать image и target
            image, target = self.transforms(image, target)

        return image, target


# --- Трансформации (пример из torchvision detection tutorial) ---
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = TF.hflip(image)
            if "boxes" in target:
                bbox = target["boxes"]
                # PIL Image width
                img_width, _ = image.size  # PIL image
                bbox[:, [0, 2]] = img_width - bbox[:, [2, 0]]  # flip xmin and xmax
                target["boxes"] = bbox
        return image, target


# Добавьте другие трансформации: Resize, Normalize и т.д.
# Normalize должна применяться после ToTensor

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())  # Сначала ToTensor
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    # Сюда можно добавить Resize, Normalize (после ToTensor)
    # Например, если вы хотите фиксированный размер:
    # transforms.insert(0, torchvision.transforms.Resize((desired_height, desired_width))) # До ToTensor
    # Но тогда и боксы надо будет масштабировать.
    # Проще использовать подход, где ToTensor идет первым, а затем нормализация.
    # Для моделей из torchvision, если вы не делаете Resize, они сами обработают разные размеры.
    return Compose(transforms)


# --- Модель ---
def get_model_instance_segmentation(num_classes):
    # Загружаем предобученную модель Faster R-CNN с ResNet50 FPN бэкбоном
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Или Faster R-CNN с MobileNetV3-Large FPN (легче и быстрее)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)

    # Получаем количество входных признаков для классификатора
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Заменяем предобученную голову на новую
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# --- Утилита для DataLoader ---
def collate_fn(batch):
    # Фильтруем None значения, если __getitem__ может их возвращать
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:  # Если все элементы были None
        return None, None
    return tuple(zip(*batch))


# --- Основная часть (демонстрация и подготовка) ---
if __name__ == "__main__":
    # --- НАСТРОЙКИ ---
    # УКАЖИТЕ ПРАВИЛЬНЫЕ ПУТИ К ВАШЕМУ ДАТАСЕТУ
    BASE_DATA_DIR = 'dog-face-recognition'  # Папка, где лежат train, test_...
    TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')

    # !!! ВАЖНО: Создайте или укажите путь к вашему файлу аннотаций !!!
    # Например, если он лежит в BASE_DATA_DIR
    ANNOTATION_FILE = os.path.join(BASE_DATA_DIR, 'train_annotations.csv')

    # Проверка наличия папок и файла аннотаций
    if not os.path.isdir(TRAIN_DIR):
        print(f"Ошибка: Директория для обучения {TRAIN_DIR} не найдена.")
        print("Пожалуйста, убедитесь, что датасет скачан и путь указан верно.")
        exit()

    # Создадим фейковый файл аннотаций для демонстрации, если его нет
    # В РЕАЛЬНОСТИ ЗДЕСЬ ДОЛЖЕН БЫТЬ ВАШ НАСТОЯЩИЙ ФАЙЛ АННОТАЦИЙ
    if not os.path.exists(ANNOTATION_FILE):
        print(f"Файл аннотаций {ANNOTATION_FILE} не найден.")
        print("Создание фейкового файла train_annotations.csv для демонстрации...")
        print("Вам НУЖНО заменить его на ваши реальные аннотации!")

        # Попробуем найти несколько изображений для фейковых аннотаций
        fake_annotations_data = {'filepath': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []}
        found_images = 0
        for root, _, files in os.walk(TRAIN_DIR):
            if found_images >= 5: break
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rel_path = os.path.relpath(os.path.join(root, file), TRAIN_DIR)
                    fake_annotations_data['filepath'].append(rel_path.replace("\\", "/"))  # для Windows
                    # Фейковые координаты (просто для примера)
                    fake_annotations_data['xmin'].append(10)
                    fake_annotations_data['ymin'].append(10)
                    fake_annotations_data['xmax'].append(100)
                    fake_annotations_data['ymax'].append(100)
                    found_images += 1
                    if found_images >= 5: break

        if found_images > 0:
            df_fake = pd.DataFrame(fake_annotations_data)
            df_fake.to_csv(ANNOTATION_FILE, index=False)
            print(f"Фейковый {ANNOTATION_FILE} создан с {found_images} записями.")
        else:
            print(f"Не удалось найти изображения в {TRAIN_DIR} для создания фейковых аннотаций.")
            print("Пожалуйста, предоставьте настоящий файл аннотаций.")
            exit()

    # --- Демонстрация работы Dataset и DataLoader ---
    try:
        print(f"Загрузка датасета из {TRAIN_DIR} с аннотациями из {ANNOTATION_FILE}")
        # Для train=True включаем аугментации (если они есть в get_transform)
        dataset = DogFaceDataset(TRAIN_DIR, ANNOTATION_FILE, transforms=get_transform(train=True))

        if len(dataset) == 0:
            print("Датасет пуст. Проверьте пути и файл аннотаций.")
            exit()

        # dataset_test = DogFaceDataset(TRAIN_DIR, ANNOTATION_FILE, transforms=get_transform(train=False)) # Для валидации

        data_loader = DataLoader(
            dataset,
            batch_size=2,  # Маленький для примера
            shuffle=True,
            num_workers=0,  # Можно увеличить, если есть CPU ядра
            collate_fn=collate_fn  # Важно для моделей детекции
        )

        print(f"Количество изображений в датасете: {len(dataset)}")

        # Посмотрим на один пример из DataLoader
        images, targets = next(iter(data_loader))

        if images is None or targets is None:
            print("Не удалось загрузить батч данных. Проверьте Dataset и collate_fn.")
            exit()

        print(f"Загружен батч: {len(images)} изображений.")
        print("Пример первого изображения в батче (тензор):", images[0].shape)
        print("Пример аннотаций для первого изображения:", targets[0])

        # Отобразим первое изображение из батча с его аннотациями
        # Для отображения нам нужно преобразовать тензор обратно в PIL Image
        # и отменить нормализацию, если она была. Сейчас нормализации нет.
        idx_to_show = 0
        img_tensor_to_show = images[idx_to_show]
        # Если есть нормализация, ее нужно обратить перед TF.to_pil_image
        # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        # img_tensor_to_show = img_tensor_to_show * std + mean
        pil_img_to_show = TF.to_pil_image(img_tensor_to_show)

        boxes_to_show = targets[idx_to_show]['boxes'].cpu().numpy()
        # Метки у нас все "мордочка", но можно добавить их для наглядности
        labels_to_show = ["dog_face"] * len(boxes_to_show)

        display_image_with_boxes_pil(pil_img_to_show, boxes_to_show, labels_to_show, title="Пример из DataLoader")

    except Exception as e:
        print(f"Произошла ошибка при инициализации Dataset/DataLoader: {e}")
        import traceback

        traceback.print_exc()
        print("Убедитесь, что файл аннотаций корректен и изображения доступны.")
        exit()

    # --- Подготовка к обучению (основной цикл обучения здесь не реализован полностью) ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Используется устройство: {device}")

    num_classes = 2  # 1 класс (мордочка) + фон
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Параметры для оптимизатора
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) # Пример

    print("\n--- Модель готова к обучению ---")
    print("Для запуска обучения вам нужно реализовать полный цикл тренировки, включая:")
    print("1. Итерацию по эпохам и батчам из data_loader.")
    print("2. Передачу изображений и целей (targets) на `device`.")
    print("3. Вызов model(images, targets) в режиме model.train().")
    print("4. Расчет общего лосса из словаря лоссов, возвращаемого моделью.")
    print("5. Обратное распространение ошибки (loss.backward()) и шаг оптимизатора (optimizer.step()).")
    print("6. (Опционально) Валидацию на тестовом/валидационном датасете с использованием model.eval().")
    print("7. Сохранение модели (torch.save).")
    print("\nПример кода для цикла обучения можно найти в официальных туториалах PyTorch по детекции объектов.")
    print("Например: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html")

    # --- Пример инференса (после обучения) ---
    # Загрузка обученной модели (замените 'path/to/your/trained_model.pth' на реальный путь)
    # model.load_state_dict(torch.load('path/to/your/trained_model.pth'))
    # model.eval()
    #
    # test_image_path = os.path.join(BASE_DATA_DIR, 'test_200_single_img', 'Adagio.jpg') # Пример
    # if os.path.exists(test_image_path):
    #     img_pil = Image.open(test_image_path).convert("RGB")
    #     img_tensor = TF.to_tensor(img_pil).to(device)
    #     # img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Если использовали при обучении
    #
    #     with torch.no_grad():
    #         prediction = model([img_tensor])
    #
    #     print("\n--- Результат инференса (пример) ---")
    #     print(prediction)
    #     pred_boxes = prediction[0]['boxes'].cpu().numpy()
    #     pred_labels_idx = prediction[0]['labels'].cpu().numpy()
    #     pred_scores = prediction[0]['scores'].cpu().numpy()
    #
    #     # Наши классы: 0 - фон, 1 - мордочка. Модель может предсказывать и фон.
    #     # Отображаем только "мордочки" с достаточной уверенностью
    #     labels_map = {1: "dog_face"}
    #     display_labels = [labels_map.get(l, "unknown") for l in pred_labels_idx]
    #
    #     display_image_with_boxes_pil(img_pil, pred_boxes, display_labels,
    #                                  title=f"Детекция на {os.path.basename(test_image_path)}",
    #                                  scores=pred_scores, score_threshold=0.5)
    # else:
    #     print(f"Тестовое изображение {test_image_path} не найдено для инференса.")