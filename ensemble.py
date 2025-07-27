import numpy as np

class EnsembleSceneDecider:
    def __init__(self, weights, decision_threshold):
        """
        Инициализирует ансамбль
        param weights: словарь с весами
        param decision_threshold: порог для принятия решения (от 0 до 1)
        """
        self.weights = weights
        self.threshold = decision_threshold
        print("Инициализация EnsembleSceneDecider:")
        print(f" - Веса: {self.weights}")
        print(f" - Порог решения: {self.threshold}")

    def decide_boundaries(self, probabilities_dict, num_shots):
        """
        Принимает словарь с вероятностями от всех экспертов и находит границы сцен
        """
        print("\n--- Ансамбль: вычисление итоговых оценок для каждой границы ---")
        num_boundaries = num_shots - 1
        
        # Проверка на корректность длин массивов
        valid_probs = {}
        for expert, probs in probabilities_dict.items():
            if len(probs) == num_boundaries:
                valid_probs[expert] = probs
            else:
                print(f"Предупреждение: некорректная длина массива от эксперта '{expert}'. Ожидалось {num_boundaries}, получено {len(probs)}. Эксперт будет проигнорирован.")

        final_scores = np.zeros(num_boundaries)
        for expert, probs in valid_probs.items():
            weight = self.weights.get(expert, 0)
            final_scores += np.array(probs) * weight
            
        max_possible_score = sum(self.weights.values())
        normalized_scores = final_scores / max_possible_score if max_possible_score > 0 else final_scores

        print("Нормализованные итоговые оценки:")
        for i, score in enumerate(normalized_scores):
            print(f"  Граница {i+1}-{i+2}: {score:.3f}")
            
        boundary_indices = [0]
        for i, score in enumerate(normalized_scores):
            if score >= self.threshold:
                boundary_indices.append(i + 1)
        
        boundary_indices.append(num_shots)
        final_boundaries = sorted(list(set(boundary_indices)))

        scenes = []
        for i in range(len(final_boundaries) - 1):
            start_shot_idx = final_boundaries[i]
            end_shot_idx = final_boundaries[i+1]
            scene_shots = list(range(start_shot_idx + 1, end_shot_idx + 1))
            scenes.append(scene_shots)
            
        return scenes




