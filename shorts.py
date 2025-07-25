import argparse
import os
import time

from imageio_ffmpeg import get_ffmpeg_exe

from VideoCutter import VideoCutter

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


if __name__ == "__main__":
    main()