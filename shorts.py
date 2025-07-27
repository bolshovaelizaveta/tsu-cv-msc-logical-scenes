import argparse
import os
import time

from imageio_ffmpeg import get_ffmpeg_exe
from VideoCutter import VideoCutter
from semantic_analyzer import SemanticSceneAnalyzer # Кластеризация + скользящее окно
from histogram_analyzer import HistogramSceneAnalyzer # Гистограмма

def main():
    parser = argparse.ArgumentParser(description="Разделение видео на логически завершенные сцены")
    parser.add_argument('--video', type=str, help= "Путь к входному файлу с видео (формат видео .mp4)")
    parser.add_argument('--output_dir', type=str, 
                        help='Путь к выходному каталогу для ' \
                        'сохранения шотов (по умолчанию: ./shots_название_видео)')
    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = args.output_dir if args.output_dir else f"./shots_{video_name}"

    print(f"Входной файл: {args.video}")
    print(f"Выходной каталог: {output_dir}")

    if not os.path.exists(args.video):
        print(f"There is no such file or directory {args.video}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Нарезаем видео на шоты. Шоты сохраняются в заданную папку
    ffmpeg_path = get_ffmpeg_exe() # необходимо для нарезки шотов
    start_time = time.time()  # Засекаем время начала обработки
    video_cutter = VideoCutter(args.video, output_dir, ffmpeg_path)    
    video_cutter.do_cutting()
    end_time = time.time()  # Засекаем время окончания

    # Выводим время выполнения в секундах с округлением
    duration = end_time - start_time
    print(f"\nВремя выполнения нарезки: {duration:.2f} секунд")

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


    # Здесь в будущем будут остальные этапы
    # Например, вызов модуля YOLO, модуля librosa и финального ансамбля...
    # print("\n--- Анализ YOLO ---")
    # ...
    # print("\n--- Финальный этап: Сборка ансамбля и нарезка сцен ---")
    # ...

if __name__ == "__main__":
    main()