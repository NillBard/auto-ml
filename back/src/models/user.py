from datetime import datetime
from typing import Dict, Any

from passlib.context import CryptContext
from sqlalchemy import TIMESTAMP, func, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method

from models.base import Base, apply_status

from s3.s3 import s3

pwd_context = CryptContext(schemes=["sha256_crypt"])

class User(Base):
    __tablename__ = "user"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(nullable=False, unique=True,
                                       deferred=True, deferred_group="sensitive")
    __password: Mapped[str] = mapped_column("password", nullable=False,
                                            deferred=True, deferred_group="sensitive")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False,
                                                 server_default=func.current_timestamp(),
                                                 deferred=True, deferred_group="date")

    @hybrid_property
    def password(self):
        return self.__password

    @password.setter
    def password(self, password):
        self.__password = pwd_context.hash(password)

    @hybrid_method
    def verify_password(self, password):
        return pwd_context.verify(password, self.__password)

class TrainProject(Base):
    __tablename__ = 'training_project'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(nullable=False)
    best_model_id: Mapped[int] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(apply_status, nullable=False, index=True, server_default='pending')
    created_by: Mapped[int] = mapped_column(ForeignKey('user.id'), index=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False,
                                                 server_default=func.current_timestamp())
    # trains: Mapped[List["TrainingConfiguration"]] = rela


class TrainingConfiguration(Base):
    __tablename__ = 'training_configurations'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # name: Mapped[str] = mapped_column(nullable=False)
    model: Mapped[str] = mapped_column(nullable=False)
    # task_type: Mapped[str] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(apply_status, nullable=False, index=True, server_default='pending')
    dataset_s3_location: Mapped[str] = mapped_column(nullable=True)
    weight_s3_location: Mapped[str] = mapped_column(nullable=True)
    onnx_s3_location: Mapped[str] = mapped_column(nullable=True)
    training_project_id: Mapped[str] = mapped_column(ForeignKey("training_project.id"), nullable=True)
    # created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False,
    #                                              server_default=func.current_timestamp())
    # created_by: Mapped[int] = mapped_column(ForeignKey('user.id'), index=True)
    training_conf: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    result_metrics: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    '''
    epoch
    time
    patience
    batch
    imgsz
    optimizer
    '''

    @hybrid_property
    def s3_dataset_url(self):
        return s3.generate_link(bucket=f"user-{self.id}", key=self.dataset_s3_location)

    @hybrid_property
    def s3_weight_url(self):
        return s3.generate_link(bucket=f"user-{self.id}", key=self.weight_s3_location)
    
    @hybrid_property
    def s3_onnx_url(self):
        return s3.generate_link(bucket=f"user-{self.id}", key=self.onnx_s3_location)