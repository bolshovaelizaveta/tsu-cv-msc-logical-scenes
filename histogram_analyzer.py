import os
import cv2
import numpy as np
from tqdm import tqdm

class HistogramSceneAnalyzer:
    def __init__(self, similarity_threshold=0.7):
        """
        Инициализирует анализатор на основе гистограмм.
        param similarity_threshold: порог схожести (от 0 до 1). Чем выше, тем строже сравнение
        """
        self.threshold = similarity_threshold
        print(f"Инициализация HistogramSceneAnalyzer.")

    def _get_shot_histogram(self, shot_path, num_frames=5):
        """
        Метод для вычисления усредненной гистограммы одного шота
        UPD: игнорируем 1 и последний кадры
        """
        cap = cv2.VideoCapture(shot_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Если шот очень короткий, берем только один центральный кадр
        if total_frames < 3:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            # Логика для одного кадра
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            cap.release()
            return hist

        # Здесь такая же логика, как и в SEMANTIC_ANALYZER
        # Выбирая кадры из диапазона от 1 до предпоследнего
        start_frame = 1
        end_frame = total_frames - 2
        
        if (end_frame - start_frame) < num_frames:
            num_frames_to_take = end_frame - start_frame + 1
        else:
            num_frames_to_take = num_frames

        frame_indices = np.linspace(start_frame, end_frame, num_frames_to_take, dtype=int)
        
        histograms = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            histograms.append(hist)
        
        cap.release()

        if not histograms: return None
        return np.mean(histograms, axis=0)

    def analyze_shots(self, shots_dir):
        """
        Принимает папку с шотами и возвращает массив вероятностей
        """
        print("\n--- Анализ по гистограммам ---")
        shot_histograms = {}
        shot_files = sorted([f for f in os.listdir(shots_dir) if f.endswith(".mp4")])

        for filename in tqdm(shot_files, desc="Вычисление гистограмм"):
            shot_path = os.path.join(shots_dir, filename)
            shot_number = int(filename.split('-')[-1].split('.')[0])
            
            histogram = self._get_shot_histogram(shot_path)
            if histogram is not None:
                shot_histograms[shot_number] = histogram
        
        if len(shot_histograms) < 2:
            return np.array([])

        break_probabilities = []
        for shot_num in range(1, len(shot_histograms)):
            prev_hist = shot_histograms.get(shot_num)
            current_hist = shot_histograms.get(shot_num + 1)
            
            if prev_hist is not None and current_hist is not None:
                similarity = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)
                prob = 1.0 - ((similarity + 1) / 2) 
                break_probabilities.append(prob)
            else:
                break_probabilities.append(0.5)

        return np.array(break_probabilities)



