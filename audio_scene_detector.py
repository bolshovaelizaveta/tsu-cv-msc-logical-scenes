import librosa
import numpy as np
import json
from moviepy.editor import VideoFileClip
import os


def detect_audio_scenes(video_path, frame_duration=1.0, threshold=0.1, save_path=None):
    print(f"Analyzing audio from: {video_path}")

    # Извлечение аудио
    clip = VideoFileClip(video_path)
    audio_path = video_path.replace('.mp4', '_temp_audio.wav')
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

    # Загрузка аудио
    y, sr = librosa.load(audio_path, sr=None)

    # Расчёт RMS энергии
    frame_length = int(sr * frame_duration)
    hop_length = frame_length
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Детектирование резких изменений энергии
    rms_diff = np.diff(rms)
    scene_changes = np.where(np.abs(rms_diff) > threshold)[0]
    scene_timestamps = [(i + 1) * frame_duration for i in scene_changes]

    print(f"Обнаружено {len(scene_timestamps)} смен сцен на секундах:")
    for t in scene_timestamps:
        print(f"  - {t:.2f} sec")

    # Сохранение в JSON, если указан путь
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(scene_timestamps, f, indent=2)
        print(f"Таймстемпы сцен сохранены в {save_path}")

    return scene_timestamps
