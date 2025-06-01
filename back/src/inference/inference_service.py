import cv2
import time
from typing import Optional
from .rtsp_reader import RTSPReader
from .detector import Detector
from .trigger_checker import TriggerChecker
from .s3_saver import S3Saver

class InferenceService:
    def __init__(self, rtsp_url: str, trigger_class: str, bucket: str, prefix: str, is_custom: bool,
                 model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        self.rtsp_reader = RTSPReader(rtsp_url)
        self.detector = Detector(model_path, is_custom)
        self.trigger_checker = TriggerChecker(trigger_class, confidence_threshold)
        self.s3_saver = S3Saver(bucket, prefix)

    def process_frame(self) -> Optional[str]:
        """
        Обрабатывает один кадр из потока
        Returns:
            Optional[str]: путь к сохраненному файлу в S3 или None
        """
        # Читаем кадр
        frame_result = self.rtsp_reader.read_frame()
        if frame_result is None:
            return None
        
        success, frame_bytes = frame_result
        if not success:
            return None

        # Детектируем объекты
        print('Детектируем объекты', flush=True)
        frame, results = self.detector.detect(frame_bytes)
        task_type = self.detector.task_type
        # Проверяем триггеры и рисуем обнаружения
        print('Проверяем триггеры и рисуем обнаружения', flush=True)
        marked_frame = self.trigger_checker.process_results(frame, results, task_type)
        if marked_frame is None:
            return None
        
        print('Конвертируем размеченный кадр в bytes', flush=True)
        # Конвертируем размеченный кадр в bytes
        success, buffer = cv2.imencode('.jpg', marked_frame)
        if not success:
            return None

        # Сохраняем в S3
        return self.s3_saver.save_frame(buffer.tobytes())

    def run(self):
        """Запускает бесконечный цикл обработки кадров"""
        # Запускаем обработку
        start_time = time.time()
        timeout = 20  # 60 секунд

        try:
            while True:
                # Проверяем, не истекло ли время
                if time.time() - start_time > timeout:
                    print("Время выполнения истекло. Остановка обработки...")
                    break
                
                result = self.process_frame()
                if result:
                    print(f"Кадр сохранен в S3: {result}")

        except KeyboardInterrupt:
            print("Остановка обработки...")
        finally:
            self.rtsp_reader.release() 

#  HOW TO USE
# from src.inference import InferenceService

# # Создаем экземпляр сервиса
# service = InferenceService(
#     rtsp_url="rtsp://your_camera_url",
#     trigger_class="person",  # класс, который мы ищем
#     bucket="your-bucket",
#     prefix="detections",
#     model_path="yolov8n.pt",
#     confidence_threshold=0.5
# )

# # Запускаем обработку
# service.run()