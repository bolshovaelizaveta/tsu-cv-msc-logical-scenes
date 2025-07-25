import os
import subprocess
import cv2
import copy

from scenedetect import ContentDetector, AdaptiveDetector
from scenedetect import SceneManager, StatsManager, open_video
from scenedetect.scene_manager import write_scene_list

class VideoCutter:
    shots_list = []
    base_video_name = ""

    def __init__(self, video_path, output_dir, ffmpeg_path):
        self.video_path = video_path
        self.base_video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_dir = output_dir
        self.ffmpeg_path = ffmpeg_path


    def detect_shots(self):
        # Анализируем видео и находим шоты

        print("Запускаем анализ видео. Это может занять несколько минут.")

        video = open_video(self.video_path)
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager=stats_manager)

        # Добавляем нужные детекторы с параметрами
        scene_manager.add_detector(ContentDetector())
        scene_manager.add_detector(AdaptiveDetector())
        
        # Запускаем детекцию шотов с прогрессом
        scene_manager.detect_scenes(video, show_progress=True)

        # Получаем список шотов
        self.shots_list = scene_manager.get_scene_list()

        # Проверяем результат и выводим сообщение
        if not self.shots_list:
            print('Шоты не найдены. Возможно, стоит попробовать уменьшить значение threshold.')
        else:
            print(f'Анализ завершен! Найдено {len(self.shots_list)} шотов.')


    def get_shots(self):
        return copy.deepcopy(self.shots_list)
    

    def print_shots(self):
        # Выводим информацию о найденных шотах
        if not self.shots_list:
            print("Шоты не найдены.")
            return

        print(f"Список из {len(self.shots_list)} найденных шотов:")

        # Проходим по списку и выводим каждый шот
        for i, shot in enumerate(self.shots_list):
            print(
                f'Шот {i+1}: '
                f'Начало {shot[0].get_timecode()} / '
                f'Конец {shot[1].get_timecode()}'
            )
    

    def export_shots_to_csv(self):
        # Экспортируем список шотов в CSV-файл
        if not self.shots_list:
            print("Шоты не найдены. Экспорт невозможен.")
            return

        # Сохраняем результат в CSV-файл
        csv_filepath = os.path.join(self.output_dir, 'shot_list.csv')

        with open(csv_filepath, 'w') as f:
            # записывает:
            # Scene Number — номер шота
            # Start Frame — номер первого кадра шота
            # Start Timecode — таймкод начала шота (часы:минуты:секунды.миллисекунды)
            # Start Time (seconds) — время начала шота в секундах
            # End Frame — номер последнего кадра шота
            # End Timecode — таймкод конца шота
            # End Time (seconds) — время конца шота в секундах
            # Length (frames) — длина шота в кадрах
            # Length (timecode) — длина шота в формате таймкода
            # Length (seconds) — длина шота в секундах
            write_scene_list(f, self.shots_list)

        print(f"Список шотов успешно сохранен в файл: {csv_filepath}")


    def cut_shots(self):
        # Нарезаем видео на основе найденных шотов
        if not self.shots_list:
            print("Шоты не найдены. Нарезка невозможна.")
            return

        print(f"Начинаем нарезку видео на {len(self.shots_list)} файлов-шотов")

        # Проходим циклом по каждому шоту в нашем списке
        for i, shot in enumerate(self.shots_list):
            # Получаем начальное и конечное время шота
            start_time = shot[0].get_timecode()
            end_time = shot[1].get_timecode()

            # Имя выходного файла
            shot_number = str(i + 1).zfill(3)
            output_filename = f"{self.base_video_name}-shot-{shot_number}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)

            # Мы ставим -ss и -to после -i
            command = [
                self.ffmpeg_path,
                '-y',
                '-i', self.video_path,
                '-ss', str(start_time),    # Начало
                '-to', str(end_time),      # Конец
                output_path                # Путь к выходному файлу
            ]

            try:
                print(f"  Нарезаем шот №{shot_number}") # ({start_time} -> {end_time})...
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"  Шот №{shot_number} успешно нарезан!")
            except subprocess.CalledProcessError as e:
                print(f"  Ошибка при нарезке шота №{shot_number}: {e.stderr.decode()}")

        print(f"Видео успешно нарезано! Проверьте папку {self.output_dir}.")


    def export_startfinish_shots_to_jpg(self):
        # Нарезаем видео на основе найденных шотов
        if not self.shots_list:
            print("Шоты не найдены. Экспорт невозможен.")
            return

        print(f"Сохраняем ключевые кадры в папку: {self.output_dir}")

        cap = cv2.VideoCapture(self.video_path)

        # Проходим по всем шотам
        for i, shot in enumerate(self.shots_list):
            # Берем кадр начала шота
            start_frame_num = shot[0].get_frames()
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)
            ret_start, start_frame = cap.read()

            # Берем кадр конца шота (за один кадр до начала следующего)
            end_frame_num = shot[1].get_frames() - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame_num)
            ret_end, end_frame = cap.read()

            if ret_start and ret_end:
                print(f"Шот №{i+1} (кадры {start_frame_num} и {end_frame_num})")
                start_filename = os.path.join(self.output_dir, f"shot_{i+1:03d}_start.jpg")
                end_filename = os.path.join(self.output_dir, f"shot_{i+1:03d}_end.jpg")
                # Сохраняем их в файлы
                cv2.imwrite(start_filename, start_frame)
                cv2.imwrite(end_filename, end_frame)
                print("-" * 30)

        # Закрываем видеофайл
        cap.release()


    def do_cutting(self):
        # Выполняем все этапы нарезки видео
        self.detect_shots()
        self.print_shots()                     # можно отключить (необходимо для отладки)
        self.export_shots_to_csv()             # можно отключить (необходимо для отладки)
        self.cut_shots()
        self.export_startfinish_shots_to_jpg() # можно отключить (необходимо для отладки)
