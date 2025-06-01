from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import ForeignKey

from models.base import Base

class RTSPLinks(Base):
    __tablename__="rtsp_links"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    rtsp_link: Mapped[str] = mapped_column(nullable=False)
    user: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)
