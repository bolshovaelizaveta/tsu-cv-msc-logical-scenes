import json

def read_labeled_data_from_json(json_name):
    # Читает данные из JSON файла
    try:
        with open(json_name, 'r', encoding="utf-8") as file:  
            data = json.load(file)      
        return data
    except FileNotFoundError:
        print(f"Ошибка: файл разметки не найден по пути {json_name}")
        return None

def convert_scenes_to_dict(scenes_list):
    """
    Превращает [[1, 7], [8, 19], ...] в {'scene_1': [1, 7], 'scene_2': [8, 19], ...}
    Это нужно для совместимости с модулем метрик
    """
    scenes_dict = {}
    for i, scene in enumerate(scenes_list):
        if scene: # Проверка на случай пустой сцены
            scenes_dict[f"scene_{i+1}"] = [scene[0], scene[-1]]
    return scenes_dict




