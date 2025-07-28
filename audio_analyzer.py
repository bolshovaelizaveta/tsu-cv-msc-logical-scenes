import numpy as np
import os

from audio_scene_detector import detect_audio_scenes # Функция для анализа аудио

class AudioSceneAnalyzer:
    def __init__(self, video_path, shots_list, tolerance_sec=1.5):
        """
        Инициализирует анализатор
        :param video_path: путь к полному исходному видеофайлу
        :param shots_list: готовый список шотов от PySceneDetect
        :param tolerance_sec: окно (в секундах) для сопоставления границ
        """
        self.video_path = video_path
        self.shots_list = shots_list
        self.tolerance = tolerance_sec
        print("Инициализация AudioSceneAnalyzer")

    def analyze_shots(self):
        """
        Анализирует полное видео и сопоставляет аудио-события с границами шотов.
        Возвращает массив вероятностей разрыва
        """
        if not self.shots_list:
            return np.array([])
            
        print("\n--- Анализ по аудиодорожке ---")
        
        # Вызываем функцию, чтобы получить временные метки разрывов
        audio_break_timestamps = detect_audio_scenes(self.video_path)
        
        if not audio_break_timestamps:
            print("Аудио-детектор не нашел границ. Возвращаем нулевые вероятности.")
            # Возвращаем массив нулей, так как разрывов нет
            return np.zeros(len(self.shots_list) - 1)

        break_probabilities = []
        # Проходим по всем границам между нашими шотами
        for i in range(len(self.shots_list) - 1):
            # Граница после i-го шота - это время конца i-го шота
            shot_boundary_time = self.shots_list[i][1].get_seconds()
            
            # Ищем, есть ли рядом "аудио-событие"
            found_match = False
            for audio_ts in audio_break_timestamps:
                if abs(shot_boundary_time - audio_ts) <= self.tolerance:
                    found_match = True
                    break # Нашли совпадение, выходим из внутреннего цикла
            
            # Если рядом с границей нашего шота есть аудио-событие, вероятность разрыва = 1.0
            prob = 1.0 if found_match else 0.0
            break_probabilities.append(prob)
            
        print("Сопоставление аудио-событий с границами шотов завершено.")
        return np.array(break_probabilities)



