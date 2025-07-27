import argparse
import os
import time

from imageio_ffmpeg import get_ffmpeg_exe
from VideoCutter import VideoCutter # Нарезка
from semantic_analyzer import SemanticSceneAnalyzer # Кластеризация + скользящее окно
from histogram_analyzer import HistogramSceneAnalyzer # Гистограмма
from ensemble import EnsembleSceneDecider # Модуль принятия решений

from pathlib import Path
import sys

# Добавляем корень проекта в PYTHONPATH
root_dir = Path(__file__).parent   # папка, где лежит example_yolo_deepsort.py
sys.path.append(str(root_dir))

from src.shot_aggregator import ShotAggregator

def main():
    parser = argparse.ArgumentParser(description="Разделение видео на логически завершенные сцены")
    parser.add_argument('--video', type=str, help= "Путь к входному файлу с видео (формат видео .mp4)")
    parser.add_argument('--shots_dir', type=str, 
                        help='Путь к каталогу с шотами, если не существует, то по умолчанию создается как: ./shots_название_видео). ' \
                             'Если не задан --video то --shots_dir указывыет на папку с уже нарезанными шотами')
    args = parser.parse_args()

    # если не передано ни одного аргумента — печатаем help и выходим
    if not args.video and not args.shots_dir:
        parser.error("Нужно указать хотя бы один из аргументов: --video или --shots_dir, или -h для вызова help")

    # Если задано видео то идет нарезка
    if (args.video): 
        if not os.path.exists(args.video):
            print(f"There is no such file or directory {args.video}")
            return
        
        if not args.shots_dir:
            shots_dir = f"./shots_{os.path.splitext(os.path.basename(args.video))[0]}"
        else:
            shots_dir = args.shots_dir

        if not os.path.exists(shots_dir):
            os.makedirs(shots_dir)
        
        # Нарезаем видео на шоты. Шоты сохраняются в заданную папку
        ffmpeg_path = get_ffmpeg_exe() # необходимо для нарезки шотов
        start_time = time.time()  # Засекаем время начала обработки
        video_cutter = VideoCutter(args.video, shots_dir, ffmpeg_path)    
        video_cutter.do_cutting()
        end_time = time.time()  # Засекаем время окончания
        duration = end_time - start_time # Выводим время выполнения в секундах с округлением
        print(f"\nВремя выполнения нарезки: {duration:.2f} секунд")

    # Если видео не задано, то берем шоты из указанной директории
    else:
        if not os.path.exists(args.shots_dir):
            print(f"There is no such directory {args.shots_dir}")
            return
        
        mp4_files = [f for f in os.listdir(args.shots_dir) if f.lower().endswith('.mp4')]
        if not mp4_files:
            print(f"There are no .mp4 files in directory {args.shots_dir}")
            return

        shots_dir = args.shots_dir
        print(f"Используем директорию с шотами: {shots_dir}")
        print(f"Количество шотов: {len(mp4_files)}")

    ### --------------------
    ### Запуск всех "экспертов" и сбор результатов
    ### --------------------

    all_probabilities = {}
    num_shots = len([f for f in os.listdir(shots_dir) if f.endswith(".mp4")])

    # Семантический анализ сцен
    print("\n--- Семантический анализ сцен (модель CLIP) ---")
    start_time = time.time()
    analyzer = SemanticSceneAnalyzer()
    semantic_probs = analyzer.analyze_shots(shots_dir)
    all_probabilities['semantic'] = semantic_probs
    duration = time.time() - start_time
    print(f"Время выполнения семантического анализа: {duration:.2f} секунд")

    # Анализ по гистограммам
    print("\n--- Анализ по цветовым гистограммам ---")
    start_time = time.time()
    hist_analyzer = HistogramSceneAnalyzer() 
    hist_probs = hist_analyzer.analyze_shots(shots_dir)
    all_probabilities['histogram'] = hist_probs
    duration = time.time() - start_time
    print(f"Время выполнения анализа по гистограммам: {duration:.2f} секунд")

    # Анализ по объектам (Yolo DeepSort)
print("\n--- Анализ шотов - Yolo и DeepSort ---")
    start_time = time.time()
    aggregator = ShotAggregator(shots_dir)
    yolo_break_probs, _ = aggregator.process() 
    all_probabilities['yolo'] = yolo_break_probs
    duration = time.time() - start_time
    print(f"Время выполнения анализа с применением Yolo: {duration:.2f} секунд")

    print("--- Принятие решения ансамблем ---")
    
    # Определяем наши веса 
    expert_weights = {
        'yolo': 3.0,        # Очень надежный эксперт
        'semantic': 2.5,    # Очень надежный эксперт
        'histogram': 0.5    # Вспомогательный
        # 'audio': 1.5,     # Вспомогательный
    }
    decision_threshold = 0.6 # Порог 60% уверенности в разрыве

    # Создаем и запускаем наш модуль принятия решений
    decider = EnsembleSceneDecider(weights=expert_weights, decision_threshold=decision_threshold)
    final_scenes = decider.decide_boundaries(all_probabilities, num_shots)

    # Результат
    print("\n--- Итоговые логические сцены, найденные ансамблем ---")
    if final_scenes:
        for i, scene in enumerate(final_scenes):
            print(f"Финальная сцена {i+1}: шоты с №{scene[0]} по №{scene[-1]}")
    else:
        print("Ансамблю не удалось определить финальные сцены.")

if __name__ == "__main__":
    main()