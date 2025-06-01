import cv2
import numpy as np
from typing import List, Dict, Optional

class TriggerChecker:
    def __init__(self, trigger_class: str, confidence_threshold: float = 0.5):
        self.trigger_class = trigger_class
        self.confidence_threshold = confidence_threshold

    def check_and_draw(self, frame: np.ndarray, detectionsold: List[Dict]) -> Optional[np.ndarray]:
        """
        Проверяет обнаружения на соответствие триггеру и рисует их на кадре
        Args:
            frame: кадр в формате numpy array
            detections: список обнаружений
        Returns:
            Optional[np.ndarray]: размеченный кадр или None, если нет триггерных обнаружений
        """
        triggered_detections = []
        
        detections = []
        for result in detectionsold:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })

        # Фильтруем обнаружения по классу и уверенности
        for det in detections:
            if (det['class_name'] == self.trigger_class and 
                det['confidence'] >= self.confidence_threshold):
                triggered_detections.append(det)

        if not triggered_detections:
            return None

        # Рисуем обнаружения на кадре
        marked_frame = frame.copy()
        for det in triggered_detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']

            # Рисуем рамку
            cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Добавляем текст с классом и уверенностью
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(marked_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return marked_frame 

    def process_results(self, frame: np.ndarray, results: List[Dict], task_type: str) -> Optional[np.ndarray]:
        """
        Обрабатывает результаты в зависимости от типа задачи и рисует их на кадре
        Args:
            frame: кадр в формате numpy array
            results: список результатов
            task_type: тип задачи ('classification', 'detection', 'segmentation', 'yolo')
        Returns:
            Optional[np.ndarray]: размеченный кадр или None, если нет триггерных обнаружений
        """
        if task_type == 'yolo':
            return self.check_and_draw(frame, results)  # Обработка для YOLO
        elif task_type == 'detection':
            # Обработка для Faster R-CNN
            boxes = results[0]['boxes'].detach().numpy()
            scores = results[0]['scores'].detach().numpy()
            labels = results[0]['labels'].detach().numpy()

            # Фильтруем результаты по уверенности
            for i in range(len(scores)):
                if scores[i] >= self.confidence_threshold:
                    x1, y1, x2, y2 = boxes[i]
                    class_name = str(labels[i])  # Преобразуем метку в строку
                    # Рисуем рамку
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Добавляем текст с классом и уверенностью
                    label = f"{class_name}: {scores[i]:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return frame
        elif task_type == 'classification':
            # Обработка для классификации
            print('Обработка для классификации', flush=True)
            for result in results:
                print('result in results', flush=True)
                label = result['class_name']
                conf = result['confidence']
                # Рисуем текст с классом
                print('Рисуем текст с классом', flush=True)
                cv2.putText(frame, f"{label}: {conf:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            return frame
        elif task_type == 'segmentation':
            # Обработка для сегментации
            print('Обработка для сегментации', flush=True)
            out = results['out']  # Получаем предсказанные классы
            scores = results.get('scores', None)
            labels = results.get('labels', None)

            # Преобразуем тензор в numpy массив и изменяем размеры
            print('Преобразуем тензор в numpy массив', flush=True)
            pred_classes = out.argmax(dim=1).squeeze(0).detach().numpy()  # Убираем размерность батча
            
            # Изменяем размер маски до размеров входного кадра
            pred_classes = cv2.resize(pred_classes, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Рисуем маски на кадре
            print('Рисуем маски на кадре', flush=True)
            for class_id in np.unique(pred_classes):
                if class_id == 0:  # Пропускаем фон
                    continue
                
                # Проверяем уверенность, если доступно
                print('Проверяем уверенность, если доступно', flush=True)
                if scores is not None and labels is not None:
                    print('Проверяем', flush=True)
                    # Убедитесь, что scores и labels имеют правильные размеры
                    if len(scores.shape) > 1:
                        score = scores[class_id].item()  # Получаем значение уверенности
                    else:
                        score = scores[class_id]  # Если scores одномерный
                    print(f'{labels[class_id]}', flush=True)
                    if score < self.confidence_threshold or labels[class_id] != self.trigger_class:
                        continue  # Пропускаем, если не соответствует триггеру или уверенности

                mask = (pred_classes == class_id).astype(np.uint8)  # Создаем бинарную маску для класса
                color = (0, 255, 0)  # Цвет для рисования маски (зеленый)

                # Применяем маску к кадру
                frame[mask == 1] = color  # Рисуем маску на кадре

                # Добавляем текст с классом и уверенностью
                if labels is not None and scores is not None:
                    label = f"Class {class_id}: {score:.2f}"
                else:
                    label = f"Class {class_id}"
                cv2.putText(frame, label, (10, 30 + class_id * 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            return frame
        else:
            raise ValueError("Неподдерживаемый тип задачи")