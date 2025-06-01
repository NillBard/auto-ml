from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from process import processor
import uuid
import asyncio
from schemas.processing import ProcessingCreate, ProcessingResponse
from db import get_database, Session
from models.processing import Processing 
from mlcore.celery_app import inference
from starlette.responses import JSONResponse

router = APIRouter()

class StreamRequest(BaseModel):
    rtsp_url: str

class StreamResponse(BaseModel):
    stream_id: str
    message: str

@router.post("/start-stream", response_model=StreamResponse)
async def start_stream(request: StreamRequest):
    """
    Запуск нового стрима
    
    Args:
        request: Данные для запуска стрима (rtsp_url)
        
    Returns:
        StreamResponse: ID стрима и статус операции
    """
    # Генерируем уникальный ID для стрима
    stream_id = str(uuid.uuid4())
    
    success = await processor.start_stream(stream_id, request.rtsp_url)
    if not success:
        raise HTTPException(status_code=400, detail="Не удалось запустить стрим")
    
    # Ждем 10 секунд перед возвратом ответа
    await asyncio.sleep(5)
    
    return StreamResponse(stream_id=stream_id, message="Стрим успешно запущен")

@router.post("/stop-stream/{stream_id}")
async def stop_stream(stream_id: str):
    """
    Остановка существующего стрима
    
    Args:
        stream_id: ID стрима для остановки
        
    Returns:
        dict: Статус операции
    """
    success = await processor.stop_stream(stream_id)
    if not success:
        raise HTTPException(status_code=400, detail="Не удалось остановить стрим")
    
    return {"message": "Стрим успешно остановлен"}

@router.get("/stream-status/{stream_id}")
async def get_stream_status(stream_id: str):
    """
    Получение статуса стрима
    
    Args:
        stream_id: Идентификатор стрима
        
    Returns:
        dict: Информация о стриме
    """
    status = await processor.get_stream_status(stream_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Стрим не найден")
    
    return status

@router.post("/processing", response_model=ProcessingResponse)
async def create_processing_record(record: ProcessingCreate, db: Session = Depends(get_database)):
    """
    Создание новой записи в таблице Processing
    """
    new_record = Processing(
        type=record.type,
        rtsp_url=record.rtsp_url,
        is_custom=record.is_custom,
        model=record.model,
        trigger_class=record.trigger_class,
        confidence_threshold=record.confidence_threshold
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    
    task = inference.delay(new_record.id)
    # Обновляем запись в БД с идентификатором задачи
    new_record.celery_task_id = task.id  # Предполагается, что поле celery_task_id существует
    db.commit()  # Сохраняем изменения

    return JSONResponse({"task_id": task.id})

from sqlalchemy import desc

@router.get("/processing", response_model=list[ProcessingResponse])
async def get_processing_records(db: Session = Depends(get_database)):
    """
    Получение всех записей из таблицы Processing
    """
    records = db.query(Processing).order_by(desc(Processing.created_at)).all()
    return [ProcessingResponse(
        id=record.id,
        status=record.status,
        type=record.type,
        rtsp_url=record.rtsp_url,
        created_at=record.created_at,
        celery_task_id=record.celery_task_id,
        is_custom=record.is_custom,
        model=record.model,
        trigger_class=record.trigger_class,
        confidence_threshold=record.confidence_threshold
    ) for record in records]

@router.get("/processing/{record_id}", response_model=ProcessingResponse)
async def get_processing_record(record_id: int, db: Session = Depends(get_database)):
    """
    Получение записи из таблицы Processing по ID
    """
    record = db.query(Processing).filter(Processing.id == record_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    
    return ProcessingResponse(
        id=record.id,
        status=record.status,
        type=record.type,
        rtsp_url=record.rtsp_url,
        created_at=record.created_at,
        celery_task_id=record.celery_task_id,
        is_custom=record.is_custom,
        model=record.model,
        trigger_class=record.trigger_class
    )

import os
import zipfile
import tempfile
from s3.s3 import s3
from fastapi.responses import FileResponse

@router.get("/download/{record_id}")
async def download_record(record_id: str, db: Session = Depends(get_database)):
    """
    Скачивание данных из бакета по ID записи
    """
    local_dir = tempfile.mkdtemp() 

    def create_zip_archive(file_paths: list, zip_name: str) -> str:
        """Создает ZIP-архив из списка файлов."""
        with zipfile.ZipFile(zip_name, 'w') as zipf:
            for file in file_paths:
                zipf.write(file, os.path.basename(file))
        return zip_name
    print('Start try')

    try:
        # Скачиваем файлы
        downloaded_files = s3.download_files_with_prefix('inference', f'{record_id}/', local_dir)
        if not downloaded_files:
            raise HTTPException(status_code=404, detail="Файлы не найдены")
        print(downloaded_files)

        # Создаем ZIP-архив
        zip_name = os.path.join(local_dir, f"detections_{record_id}.zip")
        print(f'create_zip_archive {downloaded_files} {zip_name}')
        create_zip_archive(downloaded_files, zip_name)

        return FileResponse(zip_name, media_type='application/zip', filename=os.path.basename(zip_name))
    finally:
        # Удаляем временные файлы и директорию
        for file in downloaded_files:
            os.remove(file)  # Удаляем каждый файл
        try:
            os.rmdir(local_dir)  # Удаляем пустую директорию
        except OSError as e:
            print(f"Ошибка при удалении директории: {e}")

