import cv2
import numpy as np
from typing import Optional, Dict
import threading
from queue import Queue
import time

class RTSPStream:
    def __init__(self, url: str, stream_id: str, frame_interval: float = 1.0):
        self.url = url
        self.stream_id = stream_id
        self.frame_queue = Queue(maxsize=30)  # Очередь для хранения кадров
        self.is_running = False
        self.thread = None
        self.frame_interval = frame_interval  # Интервал между кадрами в секундах
        self.last_frame_time = 0

    def start(self):
        """Запускает поток получения кадров"""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def stop(self):
        """Останавливает поток получения кадров"""
        self.is_running = False
        if self.thread:
            self.thread.join()

    def _read_frames(self):
        """Внутренний метод для чтения кадров в отдельном потоке"""
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            self.is_running = False
            return

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                continue

            current_time = time.time()
            # Проверяем, прошло ли достаточно времени с последнего сохраненного кадра
            if current_time - self.last_frame_time >= self.frame_interval:
                # Если очередь полная, удаляем старый кадр
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass

                # Добавляем новый кадр
                self.frame_queue.put(frame)
                self.last_frame_time = current_time

        cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        """Получает последний доступный кадр"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return None

class RTSPManager:
    def __init__(self):
        self.streams: Dict[str, RTSPStream] = {}

    def add_stream(self, url: str, stream_id: str) -> RTSPStream:
        """Добавляет новый RTSP поток"""
        if stream_id in self.streams:
            raise Exception(f"Поток с ID {stream_id} уже существует")

        stream = RTSPStream(url, stream_id)
        stream.start()
        self.streams[stream_id] = stream
        return stream

    def remove_stream(self, stream_id: str):
        """Удаляет RTSP поток"""
        if stream_id in self.streams:
            self.streams[stream_id].stop()
            del self.streams[stream_id]

    def get_stream(self, stream_id: str) -> Optional[RTSPStream]:
        """Получает поток по ID"""
        return self.streams.get(stream_id) 