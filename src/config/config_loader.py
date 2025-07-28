from pydantic import BaseModel, Field
import yaml
from pathlib import Path
from .scheme import AppConfig

# 3. Функция загрузки конфига
def load_config() -> AppConfig:
    # Получаем путь к директории текущего файла (config.py)
    current_dir = Path(__file__).parent

    # Формируем путь к config.yaml в той же папке
    config_path = current_dir / "default.yaml"

    # Проверяем существование файла
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file {config_path} not found")

    # Загружаем и парсим YAML
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Валидируем через Pydantic
    return AppConfig(**raw_config)
