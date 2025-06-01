import cv2
import io
from datetime import datetime
from typing import Optional
from s3.s3 import s3

class S3Saver:
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix

    def save_frame(self, frame: Optional[bytes], frame_id: str = None) -> Optional[str]:
        """
        Сохраняет кадр в S3
        Args:
            frame: кадр в формате bytes
            frame_id: уникальный идентификатор кадра (если None, генерируется автоматически)
        Returns:
            Optional[str]: путь к сохраненному файлу в S3 или None в случае ошибки
        """
        if frame is None:
            return None

        try:
            # Генерируем имя файла
            if frame_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_id = f"{timestamp}.jpg"
            
            # Формируем путь в S3
            s3_path = f"{self.prefix}/{frame_id}"

            # Создаем файловый объект в памяти
            file_obj = io.BytesIO(frame)
            
            # Загружаем в S3
            s3.upload_file(file_obj, s3_path, self.bucket)
            print(f'{s3_path} {self.bucket}')
            
            return s3_path

        except Exception as e:
            print(f"Ошибка при сохранении кадра в S3: {e}")
            return None 