from ultralytics import YOLO
import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
import contextlib
import io
import logging
from .utilites.save import save_results_to_json
import time
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
root_dir = Path(__file__).parent   # папка, где лежит shot_aggregator.py
sys.path.append(str(root_dir))

from config.config_loader import load_config
from config.scheme import ProcessingVideoConfig, TrackerConfig, AppConfig, SimilarityConfig, CompareThreshold

class ShotAggregator:
    """
    Класс для обработки видео, детекции людей, трекинга и сравнения эмбеддингов.

    Attributes:
        path_files (str): Путь к директории с видеофайлами.
        model (YOLO): Модель YOLO для детекции объектов.
        skip_frames (int): Количество пропускаемых кадров.
        sharp_threshold (float): Порог резкости для фильтрации кадров.
        confidence (float): Порог уверенности для детекции.
    """

    def __init__(self, path_files, model_path='yolov8n-seg.pt'):
        """
        Инициализация ShotAggregator.

        Args:
            path_files (str): Путь к директории с видеофайлами.
            model_path (str): Путь к файлу модели YOLO.
        """
        self.path_files = path_files
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Подавление вывода YOLO при загрузке модели
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = YOLO(model_path, verbose=False).to(device)
            logging.getLogger("ultralytics").setLevel(logging.WARNING)

        # Загрузка конфигурации
        try:
            config = load_config()
        except Exception as e:
            logging.warning(f"Ошибка загрузки конфига: {e}. Используются значения по умолчанию.")
            config = AppConfig(
                processing=ProcessingVideoConfig(),
                tracker=TrackerConfig(),
                similarity=SimilarityConfig(),
                compare=CompareThreshold(),
            )

        self.skip_frames = config.processing.skip_frames
        self.sharp_threshold = config.processing.sharp_threshold
        self.confidence = config.processing.confidence
        self.max_age = config.tracker.max_age
        self.n_init = config.tracker.n_init
        self.nn_budget = config.tracker.nn_budget
        self.similarity = config.similarity.alpha
        self.compare_threshold = config.compare.threshold

    def calculate_sharpness(self, image):
        """
        Вычисляет меру резкости изображения с помощью оператора Лапласа.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            float: Мера резкости.
        """
        if image.size == 0:
            return 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def extract_frame_features(self, frame):
        """
        Извлекает общие признаки кадра (цвет, текстура) для сравнения шотов без людей.

        Args:
            frame (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Вектор признаков кадра.
        """
        # Преобразуем в HSV цветовое пространство
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Гистограмма по каналам HSV
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

        # Признаки текстуры (LBP)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lbp = self.local_binary_pattern(gray)
        hist_lbp = cv2.calcHist([lbp], [0], None, [16], [0, 256]).flatten()

        # Объединяем все признаки
        features = np.concatenate([hist_h, hist_s, hist_v, hist_lbp])

        # Нормализуем
        features = features / (features.sum() + 1e-6)

        return features

    def local_binary_pattern(self, image, radius=3, neighbors=8):
        """
        Оптимизированная версия LBP с использованием векторизации NumPy.
        Работает в 100 раз быстрее наивной реализации.
        """
        height, width = image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)

        # Предварительно вычисляем все углы
        angles = 2 * np.pi * np.arange(neighbors) / neighbors
        x_offsets = np.round(radius * np.cos(angles)).astype(int)
        y_offsets = np.round(radius * np.sin(angles)).astype(int)

        # Создаем сетку координат
        y, x = np.mgrid[radius:height - radius, radius:width - radius]
        center_pixels = image[y, x]

        # Вычисляем значения соседей для всех точек одновременно
        neighbor_values = np.zeros((neighbors, height - 2 * radius, width - 2 * radius))
        for i in range(neighbors):
            dy, dx = y_offsets[i], x_offsets[i]
            neighbor_values[i] = image[y + dy, x + dx]

        # Сравниваем с центральным пикселем и вычисляем LBP код
        binary = (neighbor_values >= center_pixels).astype(np.uint8)
        powers = 2 ** np.arange(neighbors, dtype=np.uint8)
        lbp[y, x] = np.dot(binary.T, powers).T

        return lbp

    def process_video(self, file_path):
        """
        Обрабатывает видеофайл: детектирует людей, трекает их и сохраняет эмбеддинги.

        Args:
            file_path (str): Путь к видеофайлу.

        Returns:
            tuple: (Словарь с эмбеддингами для каждого трека, список признаков кадров)
        """
        cap = cv2.VideoCapture(file_path)
        tracker = DeepSort(max_age=self.max_age, n_init=self.n_init, nn_budget=self.nn_budget)
        saved_embeddings = defaultdict(list)
        frame_features = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // 10)  # Берем 10 кадров из видео для анализа

        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Сохраняем признаки кадра через определенные интервалы
            if frame_count % frame_interval == 0:
                frame_features.append(self.extract_frame_features(frame))

            if frame_count % (self.skip_frames + 1) != 0:
                continue

            # Подготовка размытого фона
            blurred_bg = cv2.GaussianBlur(frame, (51, 51), 0)
            masked_frame = blurred_bg.copy()

            # Подавление вывода YOLO при детекции
            results = self.model(frame, classes=[0])

            if results is not None:
                for result in results:
                    if result.masks is not None:
                        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        for mask in result.masks.data:
                            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                            mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                            combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

                        inverse_mask = cv2.bitwise_not(combined_mask)
                        foreground = cv2.bitwise_and(frame, frame, mask=combined_mask)
                        background = cv2.bitwise_and(blurred_bg, blurred_bg, mask=inverse_mask)
                        masked_frame = cv2.add(foreground, background)

                    detections = []
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        roi = frame[y1:y2, x1:x2]

                        if roi.size == 0:
                            continue

                        if (self.calculate_sharpness(roi) > self.sharp_threshold or
                                (x2 - x1) * (y2 - y1) > (frame.shape[0] * frame.shape[1]) / 8):
                            conf = float(box.conf[0].cpu().numpy())
                            if conf > self.confidence:
                                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, None])

                    tracks = tracker.update_tracks(detections, frame=masked_frame)

                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        track_id = track.track_id
                        embedding = track.features

                        if embedding is not None and track_id not in saved_embeddings:
                            saved_embeddings[track_id].append(embedding)

        cap.release()

        # Усредняем признаки кадров
        avg_frame_features = np.mean(frame_features, axis=0) if frame_features else None
        return saved_embeddings, avg_frame_features

    def compare_adjacent_videos(self, video_data_list, threshold=0.7):
        """
        Сравнивает эмбеддинги между соседними видео.

        Args:
            video_data_list (list): Список кортежей (эмбеддинги, признаки кадров) для каждого видео.
            threshold (float): Порог сходства.

        Returns:
            list: Результаты сравнения для каждой пары видео.
        """
        compare = []

        for i in range(len(video_data_list) - 1):
            emb1, frame_feat1 = video_data_list[i]
            emb2, frame_feat2 = video_data_list[i + 1]

            # Если в обоих видео есть люди - сравниваем эмбеддинги
            if emb1 and emb2:
                try:
                    emb1_list = [np.array(e).reshape(1, -1) for e in self.dict_to_list(emb1)]
                    emb2_list = [np.array(e).reshape(1, -1) for e in self.dict_to_list(emb2)]

                    emb_len = emb1_list[0].shape[1]
                    if any(e.shape[1] != emb_len for e in emb1_list + emb2_list):
                        raise ValueError("Embeddings have different dimensions")

                    sim_matrix = cosine_similarity(np.vstack(emb1_list), np.vstack(emb2_list))

                    compare.append({
                        'video_pair': (i, i + 1),
                        'pairwise_similarity': float(np.mean(sim_matrix)),
                        'match_rate': float(100 * np.mean(sim_matrix > threshold)),
                        'frame_similarity': None,
                        'comparison_type': 'people'
                    })

                except Exception as e:
                    compare.append({
                        'video_pair': (i, i + 1),
                        'pairwise_similarity': None,
                        'match_rate': None,
                        'frame_similarity': None,
                        'error': str(e),
                        'comparison_type': 'people'
                    })

            # Если в одном из видео нет людей - сравниваем признаки кадров
            elif frame_feat1 is not None and frame_feat2 is not None:
                frame_sim = cosine_similarity(frame_feat1.reshape(1, -1), frame_feat2.reshape(1, -1))[0][0]

                compare.append({
                    'video_pair': (i, i + 1),
                    'pairwise_similarity': None,
                    'match_rate': None,
                    'frame_similarity': float(frame_sim),
                    'comparison_type': 'frames'
                })

            # Если нет данных для сравнения
            else:
                compare.append({
                    'video_pair': (i, i + 1),
                    'pairwise_similarity': None,
                    'match_rate': None,
                    'frame_similarity': None,
                    'error': 'No data for comparison',
                    'comparison_type': 'none'
                })

        # Вычисляем итоговые вероятности
        probs = []
        for pair in compare:
            if pair['comparison_type'] == 'people':
                sim = pair['pairwise_similarity']
                match = pair['match_rate']

                if sim is None or match is None:
                    probs.append(0.0)
                    continue

                # Линейная комбинация similarity и match_rate
                p = self.similarity * sim + (1 - self.similarity) * (match / 100.0)
                probs.append(1.0-p)

            elif pair['comparison_type'] == 'frames':
                frame_sim = pair['frame_similarity']

                if frame_sim is None:
                    probs.append(0.0)
                    continue

                probs.append(1.0-frame_sim)

            else:
                probs.append(0.0)
        return np.array(probs), compare

    @staticmethod
    def convert_arrays_to_lists(obj):
        """
        Рекурсивно преобразует numpy массивы в списки.

        Args:
            obj: Объект для преобразования.

        Returns:
            Объект с преобразованными массивами.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [ShotAggregator.convert_arrays_to_lists(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: ShotAggregator.convert_arrays_to_lists(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def dict_to_list(data):
        """
        Преобразует словарь эмбеддингов в список.

        Args:
            data (dict): Словарь эмбеддингов.

        Returns:
            list: Список эмбеддингов.
        """
        return [item[0][0] for _, item in data.items()] if data else []

    def process(self):
        """
        Основной метод обработки всех видео в директории.

        Returns:
            tuple: (Массив вероятностей, список результатов сравнения)
        """
        files = sorted([f for f in os.listdir(self.path_files) if f.endswith(".mp4")])
        video_data = []

        for file in tqdm(files, desc=f"Обработка видео"):
            saved_embeddings, frame_features = self.process_video(os.path.join(self.path_files, file))
            data_dict = self.convert_arrays_to_lists(dict(saved_embeddings))
            video_data.append((data_dict, frame_features))

        return self.compare_adjacent_videos(video_data)

if __name__ == "__main__":
    start_time = time.time()
    aggregator = ShotAggregator("../data/shots_clock_mini")
    probs, detailed_results = aggregator.process()
    save_results_to_json(probs, detailed_results, "../result/json/", "results.json")
    print("Probabilities:", probs)
    print("Detailed results:", detailed_results)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Время выполнения: {execution_time:.4f} секунд")