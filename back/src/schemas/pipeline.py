from typing import Optional

from schemas.base import BaseModel


class TestConnection(BaseModel):
    source: str


class StartingCam(BaseModel):
    location: str
    login: str
    password: str

class ActionResponse(BaseModel):
    source_id: str
    message: Optional[str]
    status: bool
