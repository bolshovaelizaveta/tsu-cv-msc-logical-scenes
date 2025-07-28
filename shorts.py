import argparse
import os
import time
import numpy as np

from imageio_ffmpeg import get_ffmpeg_exe
from VideoCutter import VideoCutter # Нарезка
from semantic_analyzer import SemanticSceneAnalyzer # Кластеризация + скользящее окно
from histogram_analyzer import HistogramSceneAnalyzer # Гистограмма
from ensemble import EnsembleSceneDecider # Модуль принятия решений
from metric_calculator import calculate_all_metrics # Метрики
from utils import read_labeled_data_from_json, convert_scenes_to_dict
from audio_analyzer import AudioSceneAnalyzer # Анализатор аудио

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
    # Добавляем аргументы для метрик, чтобы сделать их опциональными
    parser.add_argument('--evaluate', action='store_true', help="Включить оценку по метрикам после анализа")
    parser.add_argument('--manual_labels_path', type=str, help='Путь к JSON файлу с ручной разметкой')
    
    # если не передано ни одного аргумента — печатаем help и выходим
    args = parser.parse_args()
    if not args.video and not args.shots_dir:
        parser.error("Нужно указать --video или --shots_dir.")

    # Инициализируем переменные, которые нам понадобятся
    shots_dir = None
    shots_list = []
    video_filename = ""

    # Нарезка на шоты
    if args.video:
        video_filename = os.path.basename(args.video)
        if not os.path.exists(args.video):
            print(f"Такого файла или каталога не существует {args.video}")
            return

        # Если путь к шотам не указан, генерируем его по имени видео
        if not shots_dir:
            shots_dir = f"./shots_{os.path.splitext(video_filename)[0]}"
        os.makedirs(shots_dir, exist_ok=True)

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
            
        # Создаем VideoCutter и получаем shots_list
        print(f"Нарезка/проверка шотов для видео '{args.video}' в папке '{shots_dir}'...")
        video_cutter = VideoCutter(args.video, shots_dir, get_ffmpeg_exe())
        
        if any(f.endswith('.mp4') for f in os.listdir(shots_dir)):
            print(f"Папка с шотами уже существует. Восстанавливаем информацию о сценах.")
            video_cutter.detect_shots() # Этот метод заполнит shots_list
        else:
            video_cutter.do_cutting()
        
        shots_list = video_cutter.get_shots() # Теперь shots_list будет создан в любом случае
    
    elif args.shots_dir:
        shots_dir = args.shots_dir
        if not os.path.exists(shots_dir):
            print(f"Ошибка: Папка с шотами не найдена {shots_dir}"); return
        print(f"Используем готовые шоты из папки: {shots_dir}")
        
    # Если нет видео и берем готовые шоты из папки - аудио-анализ будет пропущен
    num_shots = len([f for f in os.listdir(shots_dir) if f.lower().endswith('.mp4')])
    if num_shots < 2: print("Недостаточно шотов для анализа."); return

    ### --------------------
    ### Запуск и сбор результатов
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

    # Анализ по аудио
    # Запускаем аудио-анализ, только если есть все нужные данные, если нет - пропускаем
    if args.video and shots_list:
        print("\n--- Анализ аудиодорожки ---")
        start_time = time.time()
        audio_analyzer = AudioSceneAnalyzer(video_path=args.video, shots_list=shots_list)
        audio_probs = audio_analyzer.analyze_shots()
        all_probabilities['audio'] = audio_probs
        duration = time.time() - start_time
        print(f"Время выполнения аудио-анализа: {duration:.2f} секунд")
    else:
        print("\n--- Анализ аудиодорожки пропущен (не был предоставлен --video или не удалось получить список шотов) ---")
    
    # Определяем наши веса 
    expert_weights = {
        'yolo': 3.0,        # Очень надежный эксперт
        'semantic': 2.5,    # Очень надежный эксперт
        'histogram': 0.5    # Вспомогательный
    }

    # Добавляем вес для аудио, только если оно отработало
    if 'audio' in all_probabilities:
        expert_weights['audio'] = 1.5

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

    # Оценка по метрикам
    if args.evaluate:
        # Теперь все строки ниже имеют ОДИН уровень отступа
        if not args.manual_labels_path:
            print("\nОшибка: для оценки по метрикам необходимо указать путь к файлу с разметкой через --manual_labels_path.")
        elif not os.path.exists(args.manual_labels_path):
            print(f"\nОшибка: файл с разметкой не найден: {args.manual_labels_path}")
        else:
            print("\n\n=======================================================")
            print("--- ОЦЕНКА КАЧЕСТВА РЕЗУЛЬТАТОВ АНСАМБЛЯ ---")
            print("=======================================================")
            all_manual_labels = read_labeled_data_from_json(args.manual_labels_path)
            true_scenes_data = next((item for item in all_manual_labels if item["file_name"] == video_filename), None)
            
            if true_scenes_data:
                true_scenes_dict = true_scenes_data["scenes"]
                pred_scenes_dict = convert_scenes_to_dict(final_scenes)
                
                bounds_metric, iou_metric = calculate_all_metrics(true_scenes=true_scenes_dict, pred_scenes=pred_scenes_dict)
                
                print(f"\n--- Метрики по границам сцен ---")
                print(f"Точность (Precision): {bounds_metric['precision']:.3f}")
                print(f"Полнота (Recall):    {bounds_metric['recall']:.3f}")
                print(f"F1-мера (баланс):   {bounds_metric['f1']:.3f}")
            else:
                print(f"\nНе найдена ручная разметка для видео '{video_filename}' в файле {args.manual_labels_path}.")

if __name__ == "__main__":
    main()
