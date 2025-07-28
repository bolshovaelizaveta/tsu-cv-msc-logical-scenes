import argparse
import os
import time

from imageio_ffmpeg import get_ffmpeg_exe
from VideoCutter import VideoCutter # Нарезка
from semantic_analyzer import SemanticSceneAnalyzer # Кластеризация + скользящее окно
from histogram_analyzer import HistogramSceneAnalyzer # Гистограмма

from pathlib import Path
import sys

from src.shot_aggregator import ShotAggregator


# Добавляем корень проекта в PYTHONPATH
root_dir = Path(__file__).parent   # папка, где лежит example_yolo_deepsort.py
sys.path.append(str(root_dir))


def main():
    parser = argparse.ArgumentParser(description="Разделение видео на логически завершенные сцены")
    parser.add_argument('--video', type=str, help= "Путь к входному файлу с видео (формат видео .mp4)")
    parser.add_argument('--shots_dir', type=str, 
                        help='Путь к каталогу с шотами, если не существует, то по умолчанию создается как: ./shots_название_видео). ' \
                             'Если не задан --video то --shots_dir указывыет на папку с уже нарезанными шотами')
    parser.add_argument('--audio_only', action='store_true', help="Выполнить только аудио-анализ и нарезку")
    parser.add_argument('--output_dir', type=str, help="Путь к выходному каталогу для сохранения сцен")
    args = parser.parse_args()

    # если не передано ни одного аргумента — печатаем help и выходим
    if not args.video and not args.shots_dir:
        parser.error("Нужно указать хотя бы один из аргументов: --video или --shots_dir, или -h для вызова help")

    # Если задано видео то идет нарезка
    if (args.video): 
        if not os.path.exists(args.video):
            print(f"There is no such file or directory {args.video}")
            return

        # Проверка: если выбрана только аудио-сегментация
        if args.audio_only:
            ffmpeg_path = get_ffmpeg_exe()
            output_dir = f"./audio_scenes_{os.path.splitext(os.path.basename(args.video))[0]}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            video_cutter = VideoCutter(args.video, output_dir, ffmpeg_path)
            start_time = time.time()
            video_cutter.detect_audio_scenes()
            video_cutter.cut_audio_scenes()
            duration = time.time() - start_time
            print(f"\nАудио-сегментация завершена за {duration:.2f} секунд")
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
    ### Анализ сцен
    ### --------------------

    # Семантический анализ сцен
    print("\n--- Семантический анализ сцен (модель CLIP) ---")
    start_time = time.time()
    analyzer = SemanticSceneAnalyzer()
    logical_scenes = analyzer.analyze_shots(shots_dir)

    if logical_scenes:
        for i, scene in enumerate(logical_scenes):
            print(f"Логическая сцена {i+1}: шоты с №{scene[0]} по №{scene[-1]}")
    else:
        print("Не удалось определить логические сцены.")

    duration = time.time() - start_time
    print(f"Время выполнения семантического анализа: {duration:.2f} секунд")

    # Анализ по гистограммам
    print("\n--- Анализ по цветовым гистограммам ---")
    start_time = time.time()

    hist_analyzer = HistogramSceneAnalyzer(similarity_threshold=0.7)

    hist_probabilities = hist_analyzer.analyze_shots(shots_dir)

    if hist_probabilities.size > 0:
        print("Вероятности продолжения сцены для каждой границы:")
        for i, prob in enumerate(hist_probabilities):
            print(f"  Граница {i+1}-{i+2}: {prob:.2f}")
    else:
        print("Не удалось получить оценки по гистограммам.")

    duration = time.time() - start_time
    print(f"Время выполнения анализа по гистограммам: {duration:.2f} секунд")

    # Определение вероятности начала новой сцены между заданными шотами (Yolo DeepSort)
    print("\n--- Анализ шотов - Yolo и DeepSort ---")
    start_time = time.time()

    aggregator = ShotAggregator(shots_dir)
    yolo_probabilities, _ = aggregator.process()

    if yolo_probabilities.size > 0:
        print("Вероятности продолжения сцены для каждой границы:")
        for i, prob in enumerate(yolo_probabilities):
            print(f"  Граница {i+1}-{i+2}: {prob:.2f}")
    else:
        print("Не удалось получить оценки с использованием Yolo.")

    duration = time.time() - start_time
    print(f"Время выполнения анализа с применением Yolo: {duration:.2f} секунд")


    # Здесь в будущем будут остальные этапы
    # Например, вызов модуля YOLO, модуля librosa и финального ансамбля...
    # print("\n--- Анализ YOLO ---")
    # ...
    # print("\n--- Финальный этап: Сборка ансамбля и нарезка сцен ---")
    # ...

if __name__ == "__main__":
    main()
