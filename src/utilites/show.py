import json
import matplotlib.pyplot as plt
import os
import numpy as np

def read_and_plot_results(json_path, output_dir="plots"):
    """
    Читает JSON-файл с результатами и строит графики.

    Args:
        json_path (str): Путь к JSON-файлу.
        output_dir (str): Директория для сохранения графиков.
    """
    # 1. Чтение JSON-файла
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    probabilities = data.get("probabilities", [])
    detailed_results = data.get("detailed_results", [])

    if not probabilities and not detailed_results:
        print("Нет данных для визуализации.")
        return

    # Создаём директорию для графиков, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # 2. Построение графиков
    plt.figure(figsize=(12, 8))

    # График вероятностей
    if probabilities:
        plt.subplot(2, 1, 1)
        plt.plot(probabilities, 'b-', marker='o', label='Вероятности')
        plt.title("График вероятностей")
        plt.xlabel("Индекс")
        plt.ylabel("Значение")
        plt.grid(True)
        plt.legend()

    # График frame_similarity и pairwise_similarity
    if detailed_results:
        frame_sim = []
        pairwise_sim = []
        match_rates = []
        x_labels = []

        for i, res in enumerate(detailed_results):
            x_labels.append(f"{res['video_pair'][0]}-{res['video_pair'][1]}")
            frame_sim.append(res.get('frame_similarity', None))
            pairwise_sim.append(res.get('pairwise_similarity', None))
            match_rates.append(res.get('match_rate', None))

        plt.subplot(2, 1, 2)

        # Frame Similarity (если есть данные)
        if any(s is not None for s in frame_sim):
            plt.plot(x_labels, frame_sim, 'g-', marker='s', label='Frame Similarity')

        # Pairwise Similarity (если есть данные)
        if any(s is not None for s in pairwise_sim):
            plt.plot(x_labels, pairwise_sim, 'r-', marker='^', label='Pairwise Similarity')

        # Match Rate (если есть данные)
        if any(m is not None for m in match_rates):
            plt.plot(x_labels, match_rates, 'm--', marker='x', label='Match Rate (%)')

        plt.title("Сравнение видео (похожесть кадров/людей)")
        plt.xlabel("Пары видео")
        plt.ylabel("Значение")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()

    # Сохранение графиков
    plot_filename = os.path.join(output_dir, "results_plot.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Графики сохранены в: {plot_filename}")


def plot_probabilities_from_json(json_path, output_dir="plots", filename="probabilities_barplot.png"):
    """
    Читает JSON-файл и строит столбчатую диаграмму для probabilities.

    Args:
        json_path (str): Путь к JSON-файлу.
        output_dir (str): Директория для сохранения графика.
    """
    # 1. Чтение JSON-файла
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    probabilities = data.get("probabilities", [])

    if not probabilities:
        print("Нет данных probabilities для визуализации.")
        return

    # 2. Подготовка данных
    x = np.arange(len(probabilities))  # Индексы для оси X
    labels = [f"Shots {i+1} - {i+2}" for i in x]  # Подписи столбцов

    # 3. Настройка графика
    plt.figure(figsize=(50, 6))
    bars = plt.bar(x, probabilities, color='skyblue', edgecolor='black', alpha=0.7)

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=8)

    # 4. Оформление
    plt.title("Вероятности (столбчатая диаграмма)", fontsize=14)
    plt.xlabel("Индексы", fontsize=12)
    plt.ylabel("Значение вероятности", fontsize=12)
    plt.xticks(x, labels, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 5. Сохранение
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"График сохранён: {plot_path}")



if __name__ == "__main__":
    # Пример использования
    plot_probabilities_from_json("../../result/json/results.json",
                                 "../../result/plots/", "probabilities_barplot.png")