import os
import cv2
import time
from typing import Optional
import threading
from queue import Queue

class FrameStorage:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.frames_dir = os.path.join("/app/streams", stream_id, "frames")
        self.frame_queue = Queue(maxsize=100)
        self.is_running = False
        self.thread = None
        self.frame_counter = 0
        
        # Создаем директорию если её нет
        os.makedirs(self.frames_dir, exist_ok=True)

    def start(self):
        """Запускает поток сохранения кадров"""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._save_frames, daemon=True)
        self.thread.start()

    def stop(self):
        """Останавливает поток сохранения кадров"""
        self.is_running = False
        if self.thread:
            self.thread.join()

    def add_frame(self, frame):
        """Добавляет кадр в очередь на сохранение"""
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        self.frame_queue.put(frame)

    def _save_frames(self):
        """Внутренний метод для сохранения кадров в отдельном потоке"""
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is not None:
                    frame_path = os.path.join(self.frames_dir, f"frame_{self.frame_counter}.jpg")
                    cv2.imwrite(frame_path, frame)
                    self.frame_counter += 1
            except:
                continue

class StorageManager:
    def __init__(self):
        self.storages = {}

    def add_storage(self, stream_id: str) -> FrameStorage:
        """Создает новое хранилище для потока"""
        if stream_id in self.storages:
            raise Exception(f"Хранилище для потока {stream_id} уже существует")

        storage = FrameStorage(stream_id)
        storage.start()
        self.storages[stream_id] = storage
        return storage

    def remove_storage(self, stream_id: str):
        """Удаляет хранилище потока"""
        if stream_id in self.storages:
            self.storages[stream_id].stop()
            del self.storages[stream_id]

    def get_storage(self, stream_id: str) -> Optional[FrameStorage]:
        """Получает хранилище по ID потока"""
        return self.storages.get(stream_id) 