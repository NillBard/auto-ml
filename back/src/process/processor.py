from typing import Optional
import asyncio
from datetime import datetime
import uuid
from .rtsp import RTSPManager
from .storage import StorageManager
from .gstreamer import HLSManager
from .detection import detection_service

class Processor:
    """
    Сервис для управления процессами обработки видео
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        self.processing_threads = {}  # Словарь для хранения потоков обработки
        self.rtsp_manager = RTSPManager()
        self.storage_manager = StorageManager()
        self.hls_manager = HLSManager()

    async def start_stream(self, stream_id: str, rtsp_url: str) -> bool:
        """
        Запуск нового стрима
        
        Args:
            stream_id: Уникальный идентификатор стрима
            rtsp_url: URL RTSP потока
            
        Returns:
            bool: True если стрим успешно запущен, False в противном случае
        """
        async with self._lock:
            try:
                # Создаем RTSP поток и хранилище
                rtsp_stream = self.rtsp_manager.add_stream(rtsp_url, stream_id)
                storage = self.storage_manager.add_storage(stream_id)
                
                # Запускаем процесс сохранения кадров
                def process_frames():
                    while rtsp_stream.is_running:
                        frame = rtsp_stream.get_frame()
                        if frame is not None:
                            # Обрабатываем кадр с помощью YOLO
                            processed_frame, detections = detection_service.process_frame(frame)
                            # Сохраняем обработанный кадр
                            storage.add_frame(processed_frame)
                
                # Запускаем обработку в отдельном потоке
                import threading
                processing_thread = threading.Thread(target=process_frames, daemon=True)
                processing_thread.start()
                self.processing_threads[stream_id] = processing_thread

                # Запускаем HLS генератор
                self.hls_manager.start_generator(stream_id)
                
                return True
            except Exception as e:
                # В случае ошибки очищаем ресурсы
                if 'stream_id' in locals():
                    await self.stop_stream(stream_id)
                return False

    async def stop_stream(self, stream_id: str) -> bool:
        """
        Остановка существующего стрима
        
        Args:
            stream_id: Уникальный идентификатор стрима
            
        Returns:
            bool: True если стрим успешно остановлен, False в противном случае
        """
        async with self._lock:
            try:
                # Останавливаем RTSP поток
                self.rtsp_manager.remove_stream(stream_id)
                
                # Останавливаем хранилище
                self.storage_manager.remove_storage(stream_id)
                
                # Останавливаем HLS генератор
                self.hls_manager.stop_generator(stream_id)
                
                # Останавливаем поток обработки
                if stream_id in self.processing_threads:
                    processing_thread = self.processing_threads[stream_id]
                    if processing_thread.is_alive():
                        processing_thread.join(timeout=5.0)  # Ждем завершения потока не более 5 секунд
                    del self.processing_threads[stream_id]
                
                return True
            except:
                return False

    async def get_stream_status(self, stream_id: str) -> Optional[dict]:
        """
        Получение статуса стрима
        
        Args:
            stream_id: Уникальный идентификатор стрима
            
        Returns:
            Optional[dict]: Информация о стриме или None если стрим не найден
        """
        rtsp_stream = self.rtsp_manager.get_stream(stream_id)
        if rtsp_stream is None:
            return None
            
        hls_generator = self.hls_manager.get_generator(stream_id)
            
        return {
            "stream_id": stream_id,
            "rtsp_url": rtsp_stream.url,
            "is_running": rtsp_stream.is_running,
            "processing_thread_alive": stream_id in self.processing_threads and self.processing_threads[stream_id].is_alive(),
            "hls_generator_running": hls_generator is not None and hls_generator.is_running
        }

# Создаем глобальный экземпляр процессора
processor = Processor() 