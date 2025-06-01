from schemas.base import BaseModel
from datetime import datetime

class ProcessingCreate(BaseModel):
    type: str = "rtsp"  # Значение по умолчанию
    rtsp_url: str
    is_custom: bool
    model: str
    trigger_class: str
    confidence_threshold: float

class ProcessingResponse(BaseModel):
    id: int
    status: str
    type: str
    rtsp_url: str
    created_at: datetime
    celery_task_id: str
    is_custom: bool
    model: str
    trigger_class: str
    confidence_threshold: float
