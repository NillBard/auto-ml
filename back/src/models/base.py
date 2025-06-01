from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ENUM

Base = declarative_base()

apply_status = ENUM('pending', 'processing', 'processed', 'error', name="apply_status", metadata=Base.metadata)

