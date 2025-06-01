import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
from inference.model_tailor import Tailor

class Detector:
    def __init__(self, model_path: str, is_custom: bool):
        self.tailor = Tailor(model_path, is_custom)
        self.model = self.tailor.model
        self.task_type = self.tailor.task_type

    def detect(self, frame_bytes: bytes) -> Tuple[np.ndarray, List[Dict]]:
        """
        Обработка кадра с помощью YOLO
        Args:
            frame_bytes: кадр в формате bytes
        Returns:
            Tuple[np.ndarray, List[Dict]]: (кадр в формате numpy array, список обнаружений)
        """
        # Конвертируем bytes в numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Получаем предсказания от модели
        print('Получаем предсказания от модели', flush=True)
        results = self.tailor.predict(frame)
        print('Получили предсказания от модели', flush=True)

        return frame, results 