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
        UPD: убираем 1 и последний кадр из шота для гарантии "чистоты" шота
        """
        cap = cv2.VideoCapture(shot_path)
        if not cap.isOpened(): return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Если шот очень короткий (меньше 3 кадров), просто берем средний
        if total_frames < 3:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            if not ret: 
                cap.release()
                return None
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                features = self.clip_model.get_image_features(**inputs)
            cap.release()
            return features.cpu().squeeze().numpy()

        start_frame = 1
        end_frame = total_frames - 2

        # Если шот короткий, но больше 3 кадров, берем меньше 5 кадров
        if (end_frame - start_frame) < num_frames:
            num_frames_to_take = end_frame - start_frame + 1
        else:
            num_frames_to_take = num_frames

        frame_indices = np.linspace(start_frame, end_frame, num_frames_to_take, dtype=int)
        
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
        Принимает папку с шотами и возвращает массив вероятностей разрыва
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
        
        # Проверка, если эмбеддингов слишком мало для анализа
        if len(shot_embeddings) < 2:
            print("Не удалось извлечь достаточно эмбеддингов для анализа.")
            return np.array([])

        print(f"\n--- Анализ эмбеддингов (кластеризация + скользящее окно) ---")
        shot_numbers = sorted(shot_embeddings.keys())
        all_embeddings = np.array([shot_embeddings[num] for num in shot_numbers])

        # Эти параметры показывали лучший результат
        PERCENTILE = 90
        
        distances_matrix = pairwise_distances(all_embeddings, metric='cosine')
        upper_triangle_indices = np.triu_indices_from(distances_matrix, k=1)
        
        if len(upper_triangle_indices[0]) == 0:
             return np.array([])

        unique_distances = distances_matrix[upper_triangle_indices]
        distance_threshold = np.percentile(unique_distances, PERCENTILE)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='complete',
            metric='cosine'
        )
        cluster_labels = clustering.fit_predict(all_embeddings)

        # Вместо поиска готовых сцен, мы теперь вычисляем массив вероятностей для каждой границы
        
        print("Вычисление вероятностей разрыва сцены...")
        
        break_probabilities = []
        # Мы используем те же параметры, которые показали лучший результат
        WINDOW_SIZE = 2
        
        for i in range(len(cluster_labels) - 1): # Проходим по всем границам между шотами
            # Окно ДО и ПОСЛЕ текущей границы
            window_before = set(cluster_labels[max(0, i - WINDOW_SIZE + 1) : i + 1])
            window_after = set(cluster_labels[i + 1 : min(len(cluster_labels), i + 1 + WINDOW_SIZE)])
            
            intersection = len(window_before.intersection(window_after))
            union = len(window_before.union(window_after))
            jaccard_similarity = intersection / union if union != 0 else 0
            
            # Превращаем "похожесть" в "вероятность разрыва" (1.0 - схожесть)
            break_prob = 1.0 - jaccard_similarity
            break_probabilities.append(break_prob)
            
        # Возвращаем массив с "оценками" для каждой границы
        return np.array(break_probabilities)




