import json
import os
import numpy as np


def save_results_to_json(probabilities, detailed_results, output_dir, filename="results.json"):
    """
    Сохраняет probabilities и detailed_results в JSON-файл.
    Автоматически конвертирует numpy-массивы и скаляры в JSON-совместимые типы.

    Args:
        probabilities (np.ndarray или list): Массив вероятностей.
        detailed_results (list): Массив с детализированными результатами.
        output_dir (str): Директория для сохранения файла.
        filename (str, optional): Имя файла. По умолчанию "results.json".

    Returns:
        str: Полный путь к сохранённому файлу.
    """
    # Конвертируем numpy-массивы в списки
    if isinstance(probabilities, np.ndarray):
        probabilities = probabilities.tolist()

    # Функция для рекурсивной конвертации numpy-типов
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        return obj

    # Применяем конвертацию ко всем данным
    detailed_results = convert_numpy(detailed_results)

    # Создаём директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Формируем данные для сохранения
    data = {
        "probabilities": probabilities,
        "detailed_results": detailed_results
    }

    # Полный путь к файлу
    filepath = os.path.join(output_dir, filename)

    # Сохраняем в JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return filepath