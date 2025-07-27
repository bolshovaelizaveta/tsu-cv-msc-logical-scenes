import json  


def read_labeled_data_from_json(json_name):
    with open(json_name, encoding="utf-8") as file:  
        data = json.load(file)      
    return data


def write_labeled_data_to_json(json_name,  data):
    with open(json_name, 'w', encoding="utf-8") as file:
        json.dump(data, file)


def get_left_bounds(scenes):
    # Получаем все левые границы сцен
    return sorted([bounds[0] for bounds in scenes.values()])


def calculate_precision_recall_f1(TP, len_true, len_pred):
    # Рассчитываем метрики Precision, Recall, F1
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # f1 = 2 * (precision * recall) / (precision + recall)

    FP = len_pred - TP
    FN = len_true - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def boundaries_metrics(true_scenes, pred_scenes, tolerance=3):
    # Precision, Recall, F1 на границах сцен
    # определяются по близости к реальным границам.
    # Порог допустимого отклонения ± 3 шота по умолчанию
    # TP = число правильно предсказанных границ (в пределах допустимого окна)
    # FP = число лишних предсказанных границ
    # FN = число пропущенных границ
    # Precision = сколько из предсказанных границ совпали с реальными
    # Recall = сколько из реальных границ были найдены
    # F1 = гармоническое среднее между Precision и Recall

    # Берем левую границу т.к. для обоих границ метрики должны совпасть, 
    # а левая более интерпретируемая и понятная - начало сцены

    matched = set()

    left_bounds_true = get_left_bounds(true_scenes)
    left_bounds_pred = get_left_bounds(pred_scenes)
    print(f"Границы ручной разметки слева: {left_bounds_true}")
    print(f"Границы предсказанной разметки слева: {left_bounds_pred}")    
    
    TP = 0
    # Сравниваем предсказанные и истинные границы
    for pred_bound in left_bounds_pred:
        for i, true_bound in enumerate(left_bounds_true):

            if i in matched: # Если граница уже была найдена пропускаем
                continue
            # Если предсказанная граница близка к истинной (в пределах tolerance), увеличиваем TP
            if abs(pred_bound - true_bound) <= tolerance:
                TP += 1
                matched.add(i) # Запоминаем, что граница найдена
                break
    
    # рассчитываем Precision, Recall, F1 
    result = calculate_precision_recall_f1(TP, len(left_bounds_true), len(left_bounds_pred))

    return result


def calculate_iou_metrics(true_scene, pred_scene):
    # Вычисляем IoU, IoM, IoA
    # IoU — Intersection over Union
    # IoU = intersection(true_scene, pred_scene) / union(true_scene, pred_scene)
    # Насколько точно предсказанный отрезок совпадает с истинным по времени.
    # Если предсказанная сцена точно повторяет границы ручной разметки, IoU = 1.
    # Если сцены лишь частично пересекаются, IoU будет меньше.
    # Строгая метрика, учитывает не только наличие пересечения, но и насколько предсказание близко к истинным границам.

    # IoM — Intersection over Minimum
    # IoM = intersection(true_scene, pred_scene) / min(len(true_scene), len(pred_scene))
    # Насколько полностью меньшая из двух сцен покрыта пересечением.
    # Если меньшая сцена полностью входит в большую, IoM = 1, даже если вторая сцена значительно длиннее.
    # Мягкая метрика, хорошо показывает, что хотя бы одна сцена полностью захвачена другой.

    # IoA — Intersection over Maximum
    # IoA = intersection(true_scene, pred_scene) / max(len(true_scene), len(pred_scene))
    # Насколько пересечение покрывает большую сцену из двух
    # Если предсказание значительно длиннее, IoA будет ниже.
    # Строгая метрика, которая штрафует за "перепредсказания" — когда модель ставит сцены длиннее реальных.

    start_true_scene, end_true_scene = true_scene
    start_pred_scene, end_pred_scene = pred_scene

    intersect_start = max(start_true_scene, start_pred_scene)  # Начало пересечения
    intersect_end = min(end_true_scene, end_pred_scene)        # Конец пересечения
    intersection = max(0, intersect_end - intersect_start + 1) # max(0, ...) если интервалы не пересекаются (тогда длина пересечения 0)

    len_true_scene = end_true_scene - start_true_scene + 1 # +1, чтобы считать включительно начальный и конечный кадр
    len_pred_scene = end_pred_scene - start_pred_scene + 1 # +1, чтобы считать включительно начальный и конечный кадр

    # Объединение — это сумма длин сцен минус длина пересечения
    union = len_true_scene + len_pred_scene - intersection
    iou = intersection / union if union > 0 else 0
    iom = intersection / min(len_true_scene, len_pred_scene) if min(len_true_scene, len_pred_scene) > 0 else 0
    ioa = intersection / max(len_true_scene, len_pred_scene) if max(len_true_scene, len_pred_scene) > 0 else 0

    return iou, iom, ioa


def intersection_over_union_metrics(true_scenes, pred_scenes, threshold=0.5):
    # Рассчитываем метрики IoU, IoM, IoA
    # Метрики оценки качества сцены основаны на сравнении интервалов времени, соответствующих сценам, 
    # по трём ключевым показателям: IoU, IoM и IoA.
    # IoU отражает точное совпадение границ сцен, 
    #       требуя хорошего наложения и минимального лишнего времени.
    # IoM служит более мягкой метрикой, показывающей,
    #       насколько полностью меньшая сцена покрывается другой, позволяя оценивать частичные попадания.
    # IoA помогает выявить случаи перепредсказания, 
    #       когда сцены модели выходят за пределы реальных сцен, штрафуя избыточно длинные предсказания.
    # threshold = 0.5 - сцены считаются совпавшими, если они перекрываются хотя бы наполовину.

    # Структура для сравнения результата
    # TP — число совпавших сцен по заданному порогу
    # matched_* нужны, чтобы не сопоставлять одну сцену несколько раз
    true_scenes = list(true_scenes.values())
    pred_scenes = list(pred_scenes.values())

    metrics = {
    'iou': {'TP': 0, 'matched_true': set(), 'matched_pred': set()},
    'iom': {'TP': 0, 'matched_true': set(), 'matched_pred': set()},
    'ioa': {'TP': 0, 'matched_true': set(), 'matched_pred': set()},
    }
    
    for metric_name in metrics.keys():
        matched_true = set()
        matched_pred = set()

        for i, true_scene in enumerate(true_scenes):
            best_score = 0 # лучшее значение метрики найденное для текущей true сцены.
            best_j = -1    # индекс наилучшей предсказанной сцены, которая максимально перекрывается с текущей true сценой. (-1 - ничего не найдено)
            
            # Из всех предсказанных сцен, ищется сцену из предсказания, 
            # которая лучше всего перекрывается с true сценой, и помечается как сопоставленная.
            for j, pred_scene in enumerate(pred_scenes):
                if j in matched_pred:
                    continue  # eсли сцена уже сопоставлена, пропускаем
                iou, iom, ioa = calculate_iou_metrics(true_scene, pred_scene)
                score = {'iou': iou, 'iom': iom, 'ioa': ioa}[metric_name]

                # Ищем предсказанную сцену с максимальным score, который выше порога threshold
                if score >= threshold and score > best_score:
                    best_score = score
                    best_j = j

            # Если условие верно, то нашлось хорошее совпадение, вычисляется TP и добавляется индекс в matched_pred
            if best_j != -1: 
                matched_true.add(i)
                matched_pred.add(best_j)

        metrics[metric_name]['TP'] = len(matched_true)
        metrics[metric_name]['matched_true'] = matched_true
        metrics[metric_name]['matched_pred'] = matched_pred

    # рассчитываем Precision, Recall, F1
    # TP — количество совпавших сцен
    # FN — сколько true сцен не нашли совпадений
    # FP — сколько предсказанных сцен не совпало ни с одной true сценой
    # Precision — доля корректных предсказаний среди всех предсказаний
    # Recall — доля найденных true сцен среди всех истинных
    # F1 — гармоническое среднее Precision и Recall, общая оценка качества
    results = {}
    for metric_name, data in metrics.items():
        results[metric_name] = calculate_precision_recall_f1(data['TP'], len(true_scenes), len(pred_scenes))

    return results
    

def calculate_all_metrics(true_scenes, pred_scenes):
    # Если сцены сильно различаются по длине (короткие vs длинные), IoU дадут более реалистичную оценку.
    # Если же цель — оценить точность "границ", то Precision/Recall на границах сцен — более честный и интерпретируемый способ.

    # На вход подаются словари такого вида - 
    # {
    #   'scene_1': [1, 42],     # 'сцена_номер': [номер стартового шота, номер финишного шота]
    #   'scene_2': [43, 59], 
    #   'scene_3': [59, 120]
    # }  

    bound_metrics = boundaries_metrics(true_scenes, pred_scenes)
    iou_metrics = intersection_over_union_metrics(true_scenes, pred_scenes)

    return bound_metrics, iou_metrics


if __name__ == '__main__':
    # проверка на тестовом mockup_labeled_data.json заданном руками
    film_labels_manual = read_labeled_data_from_json('./video/manual_labeled_data.json')
    film_labels_test = read_labeled_data_from_json('mockup_labeled_data.json')    

    clock_mini_manual = next(
    (film for film in film_labels_manual if film["file_name"] == "clock_mini.mp4"),
        None
    )

    clock_mini_ideal = film_labels_test[0]
    clock_mini_one_scene = film_labels_test[1]
    clock_mini_too_often = film_labels_test[2]
    clock_mini_too_rare = film_labels_test[3]
    clock_mini_intersect = film_labels_test[4]

    print("===============================================")
    print("============clock_mini_ideal.mp4===============")
    print("===============================================")
    bounds_metric, iou_metric = calculate_all_metrics(
        true_scenes=clock_mini_manual["scenes"], 
        pred_scenes=clock_mini_ideal["scenes"] # ideal
        )
    print(f"Метрики границ:")
    print(f"precision = {bounds_metric['precision']:.2f}\nrecall = {bounds_metric['recall']:.2f},\nf1 = {bounds_metric['f1']:.2f}")
    print("-----------------------------------------------")
    print(f"Метрики IoU:")
    for metric_key, metric_value in iou_metric.items():
        print(f"{metric_key} = \nprecision = {metric_value['precision']:.2f}\nrecall = {metric_value['recall']:.2f},\nf1 = {metric_value['f1']:.2f}")
        print("---")

    print("===============================================")
    print("==========clock_mini_one_scene.mp4=============")
    print("===============================================")
    bounds_metric, iou_metric = calculate_all_metrics(
        true_scenes=clock_mini_manual["scenes"], 
        pred_scenes=clock_mini_one_scene["scenes"] # one_scene
        )
    print(f"Метрики границ:")
    print(f"precision = {bounds_metric['precision']:.2f}\nrecall = {bounds_metric['recall']:.2f},\nf1 = {bounds_metric['f1']:.2f}")
    print("-----------------------------------------------")
    print(f"Метрики IoU:")
    for metric_key, metric_value in iou_metric.items():
        print(f"{metric_key} = \nprecision = {metric_value['precision']:.2f}\nrecall = {metric_value['recall']:.2f},\nf1 = {metric_value['f1']:.2f}")
        print("---")
    
    print("===============================================")
    print("============clock_mini_too_often.mp4===========")
    print("===============================================")
    bounds_metric, iou_metric = calculate_all_metrics(
        true_scenes=clock_mini_manual["scenes"], 
        pred_scenes=clock_mini_too_often["scenes"] # too_often
        )
    print(f"Метрики границ:")
    print(f"precision = {bounds_metric['precision']:.2f}\nrecall = {bounds_metric['recall']:.2f},\nf1 = {bounds_metric['f1']:.2f}")
    print("-----------------------------------------------")
    print(f"Метрики IoU:")
    for metric_key, metric_value in iou_metric.items():
        print(f"{metric_key} = \nprecision = {metric_value['precision']:.2f}\nrecall = {metric_value['recall']:.2f},\nf1 = {metric_value['f1']:.2f}")
        print("---")

    print("===============================================")
    print("============clock_mini_too_rare.mp4============")
    print("===============================================")
    bounds_metric, iou_metric = calculate_all_metrics(
        true_scenes=clock_mini_manual["scenes"], 
        pred_scenes=clock_mini_too_rare["scenes"] # too_rare
        )
    print(f"Метрики границ:")
    print(f"precision = {bounds_metric['precision']:.2f}\nrecall = {bounds_metric['recall']:.2f},\nf1 = {bounds_metric['f1']:.2f}")
    print("-----------------------------------------------")
    print(f"Метрики IoU:")
    for metric_key, metric_value in iou_metric.items():
        print(f"{metric_key} = \nprecision = {metric_value['precision']:.2f}\nrecall = {metric_value['recall']:.2f},\nf1 = {metric_value['f1']:.2f}")
        print("---")

    print("===============================================")
    print("============clock_mini_intersect.mp4===========")
    print("===============================================")
    bounds_metric, iou_metric = calculate_all_metrics(
        true_scenes=clock_mini_manual["scenes"], 
        pred_scenes=clock_mini_intersect["scenes"] # intersect
        )
    print(f"Метрики границ:")
    print(f"precision = {bounds_metric['precision']:.2f}\nrecall = {bounds_metric['recall']:.2f},\nf1 = {bounds_metric['f1']:.2f}")
    print("-----------------------------------------------")
    print(f"Метрики IoU:")
    for metric_key, metric_value in iou_metric.items():
        print(f"{metric_key} = \nprecision = {metric_value['precision']:.2f}\nrecall = {metric_value['recall']:.2f},\nf1 = {metric_value['f1']:.2f}")
        print("---")
