from schemas.base import BaseModel


class ProjectsSchema(BaseModel):
    id: int
    name: str
    created_date: str
    status: str