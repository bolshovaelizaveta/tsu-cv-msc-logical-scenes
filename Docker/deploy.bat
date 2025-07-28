@echo off
echo =================================
echo KION Video Segmentation Deploy
echo Ensemble System: YOLO + CLIP + Histograms
echo =================================

REM Проверка установки Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: Docker не установлен или недоступен
    echo Установите Docker Desktop с https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Проверка docker-compose
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: docker-compose не найден
    pause
    exit /b 1
)

echo Docker обнаружен. Продолжаем установку...

REM Создание директорий
echo Создание рабочих директорий...
if not exist "data" mkdir data
if not exist "data\input" mkdir data\input
if not exist "data\output" mkdir data\output
if not exist "output" mkdir output
if not exist "result" mkdir result
if not exist "result\json" mkdir result\json
if not exist "result\plots" mkdir result\plots

echo Директории созданы:
echo - data\input  (поместите сюда входные видео)
echo - data\output (результаты обработки)
echo - result\json (JSON результаты анализа)
echo - result\plots (графики и визуализация)

echo.
echo Начинаем сборку Docker образа...
echo ВНИМАНИЕ: Первая сборка займет 20-40 минут!
echo Загружаются: PyTorch + CLIP + YOLOv8 + DeepSort
docker-compose build
if %errorlevel% neq 0 (
    echo ОШИБКА: Сборка образа неудачна
    pause
    exit /b 1
)

echo.
echo Тестируем систему на встроенном видео...
docker-compose run --rm video-segmentation \
  --video /app/video/sheldon.mp4 \
  --shots_dir /data/output/system_test
if %errorlevel% neq 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Тестовый запуск неудачен, но образ собран
)

echo.
echo =================================
echo Установка завершена успешно!
echo =================================
echo.
echo СИСТЕМА: Ансамбль из 3 экспертов
echo - YOLO + DeepSort (вес 3.0): трекинг людей и объектов
echo - CLIP семантика (вес 2.5): глубокий анализ содержания
echo - Гистограммы (вес 0.5): цветовая валидация
echo - Порог принятия решения: 60%%
echo.
echo Команды для запуска:
echo.
echo 1. Полный пайплайн (видео + ансамбль):
echo    Поместите видео в папку data\input\
echo    docker-compose run --rm video-segmentation --video /data/input/ваше_видео.mp4 --shots_dir /data/output/analysis
echo.
echo 2. Анализ готовых шотов:
echo    docker-compose run --rm video-segmentation --shots_dir /data/output/existing_shots
echo.
echo 3. Тестирование системы:
echo    docker-compose run --rm video-segmentation --video /app/video/sheldon.mp4 --shots_dir /data/output/test
echo.
echo РЕЗУЛЬТАТЫ:
echo - Финальные сцены: в консоли
echo - Детальные метрики: result\json\
echo - Графики: result\plots\
echo.
echo Для просмотра логов: docker-compose logs -f
echo.
pause