import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple

class DetectionService:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Инициализация сервиса детекции объектов
        
        Args:
            model_path: Путь к модели YOLOv8 или название предобученной модели
        """
        self.model = YOLO(model_path)
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Обработка кадра с помощью YOLOv8
        
        Args:
            frame: Входной кадр в формате numpy array
            
        Returns:
            Tuple[np.ndarray, list]: Обработанный кадр с нанесенными обнаружениями и список обнаружений
        """
        # Получаем предсказания от модели
        results = self.model(frame)
        
        # Получаем первый результат (так как обрабатываем один кадр)
        result = results[0]
        
        # Рисуем обнаружения на кадре
        annotated_frame = result.plot()
        
        # Собираем информацию о обнаружениях
        detections = []
        for box in result.boxes:
            detection = {
                'class': result.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            }
            detections.append(detection)
            
        return annotated_frame, detections

# Создаем глобальный экземпляр сервиса детекции
detection_service = DetectionService() 