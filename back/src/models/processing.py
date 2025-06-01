from datetime import datetime
from sqlalchemy import ForeignKey, TIMESTAMP, String, func, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from models.base import Base, apply_status
from sqlalchemy.dialects.postgresql import UUID
import uuid

class Processing(Base):
    __tablename__ = 'processing'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(apply_status, nullable=False, index=True, server_default='pending')
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.current_timestamp())
    type: Mapped[str] = mapped_column(String, nullable=False, default='rtsp')  # Поле type
    rtsp_url: Mapped[str] = mapped_column(String, nullable=False)  # Поле rtsp_url
    s3_prefix: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4)  # Поле s3_prefix
    celery_task_id: Mapped[str] = mapped_column(String, nullable=True)
    is_custom: Mapped[bool] = mapped_column(Boolean, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=True) # Название модели или ссылка если есть чек
    trigger_class: Mapped[str] = mapped_column(String, nullable=True)
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=True)