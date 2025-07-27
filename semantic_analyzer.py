import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances

class SemanticSceneAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print("Инициализация SemanticSceneAnalyzer: загрузка модели CLIP...")
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("Модель CLIP успешно загружена.")

    def _get_embedding_from_shot(self, shot_path, num_frames=5):
        """
        Метод для получения усредненного эмбеддинга из одного шота
        """
        cap = cv2.VideoCapture(shot_path)
        if not cap.isOpened(): return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames: num_frames = total_frames # Если шот короткий

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        embeddings = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            with torch.no_grad():
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                features = self.clip_model.get_image_features(**inputs)
                embeddings.append(features.cpu().squeeze().numpy())

        cap.release()
        
        if not embeddings: return None
        return np.mean(embeddings, axis=0)

    def analyze_shots(self, shots_dir):
        """
        Принимает папку с шотами и возвращает список границ сцен
        """
        print("\n--- Извлечение эмбеддингов (метод 'среднее из 5') ---")
        shot_embeddings = {}
        shot_files = sorted([f for f in os.listdir(shots_dir) if f.endswith(".mp4")])

        for filename in tqdm(shot_files, desc="Получение эмбеддингов"):
            shot_path = os.path.join(shots_dir, filename)
            shot_number = int(filename.split('-')[-1].split('.')[0])
            
            embedding = self._get_embedding_from_shot(shot_path)
            if embedding is not None:
                shot_embeddings[shot_number] = embedding
        
        if not shot_embeddings:
            print("Не удалось извлечь эмбеддинги.")
            return []

        print(f"\n--- Анализ эмбеддингов (Кластеризация + Скользящее окно) ---")
        shot_numbers = sorted(shot_embeddings.keys())
        all_embeddings = np.array([shot_embeddings[num] for num in shot_numbers])

        # Эти параметры показывали лучший результат
        PERCENTILE = 90
        WINDOW_SIZE = 2
        SIMILARITY_THRESHOLD_JACCARD = 0.5
        
        distances_matrix = pairwise_distances(all_embeddings, metric='cosine')
        # Исключаем диагональ, чтобы не учитывать расстояние от шота до самого себя
        upper_triangle_indices = np.triu_indices_from(distances_matrix, k=1)
        
        if len(upper_triangle_indices[0]) == 0: # Если шотов меньше 2
             return [list(range(1, len(shot_numbers) + 1))]

        unique_distances = distances_matrix[upper_triangle_indices]
        distance_threshold = np.percentile(unique_distances, PERCENTILE)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='complete',
            metric='cosine'
        )
        cluster_labels = clustering.fit_predict(all_embeddings)
        
        scene_boundaries = [0]
        for i in range(len(cluster_labels) - WINDOW_SIZE):
            window_before = set(cluster_labels[i : i + WINDOW_SIZE])
            window_after = set(cluster_labels[i + 1 : i + 1 + WINDOW_SIZE])
            intersection = len(window_before.intersection(window_after))
            union = len(window_before.union(window_after))
            jaccard = intersection / union if union != 0 else 0
            if jaccard < SIMILARITY_THRESHOLD_JACCARD:
                if i + WINDOW_SIZE > scene_boundaries[-1] + (WINDOW_SIZE / 2):
                     scene_boundaries.append(i + WINDOW_SIZE)
        
        scene_boundaries.append(len(shot_numbers))
        final_boundaries = sorted(list(set(scene_boundaries)))
        
        # Формируем и возвращаем список сцен (каждая сцена - список номеров шотов)
        scenes = []
        for i in range(len(final_boundaries) - 1):
            start_shot_idx = final_boundaries[i]
            end_shot_idx = final_boundaries[i+1]
            # Номера шотов начинаются с 1
            scene_shots = list(range(shot_numbers[start_shot_idx], shot_numbers[end_shot_idx-1] + 1))
            scenes.append(scene_shots)
            
        return scenes




