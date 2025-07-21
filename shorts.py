import argparse
import os

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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Нарезаем видео на шоты. Шоты сохраняются в заданную папку
    ffmpeg_path = get_ffmpeg_exe() # необходимо для нарезки шотов
    video_cutter = VideoCutter(args.video, output_dir, ffmpeg_path)
    video_cutter.do_cutting()


if __name__ == "__main__":
    main()