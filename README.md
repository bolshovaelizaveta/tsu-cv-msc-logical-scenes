# tsu-cv-msc-logical-scenes

ТГУ | Магистратура "Компьютерное зрение и нейронные сети" | 2025  Этот проект представляет собой прототип системы для автоматического разбиения видео на логически завершенные сцены. Работа выполнена в рамках хакатона по курсу.

## Запуск скрипта

### Пример

---

Автоматическое создание папки:

```shell
python shorts.py --video input.mp4
```

Задаем папку:

```shell
python shorts.py --video input.mp4 --output_dir ./my_folder_name
```

Для видео из папки video:

```shell
python shorts.py --video ./video/sheldon.mp4
```

### Help

---

`python shorts.py -h`

```shell
usage: shorts.py [-h] [--video VIDEO] [--output_dir OUTPUT_DIR]

Разделение видео на логически завершенные сцены

options:
  -h, --help            show this help message and exit
  --video VIDEO         Путь к входному файлу с видео (формат видео .mp4)
  --output_dir OUTPUT_DIR
                        Путь к выходному каталогу для сохранения сцен (по умолчанию: ./название_видео_scenes)
```
