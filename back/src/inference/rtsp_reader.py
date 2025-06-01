import cv2
import time
from typing import Optional, Tuple

class RTSPReader:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.last_frame_time = 0
        self.frame_interval = 1.0  # интервал между кадрами в секундах

    def connect(self) -> bool:
        """Подключение к RTSP потоку"""
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            return self.cap.isOpened()
        except Exception as e:
            print(f"Ошибка при подключении к RTSP потоку: {e}")
            return False

    def read_frame(self) -> Optional[Tuple[bool, bytes]]:
        """
        Чтение кадра из потока с интервалом в 1 секунду
        Returns:
            Tuple[bool, bytes]: (успех чтения, кадр в формате bytes)
        """
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return None

        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        self.last_frame_time = current_time
        
        # Конвертируем кадр в bytes
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            return None

        return True, buffer.tobytes()

    def release(self):
        """Освобождение ресурсов"""
        if self.cap is not None:
            self.cap.release() 