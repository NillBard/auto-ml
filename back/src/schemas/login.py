from pydantic import EmailStr

from schemas.base import BaseModel


class AccountCredentials(BaseModel):
    login: EmailStr
    password: str


class Refresh(BaseModel):
    refresh: str


class UserResponse(BaseModel):
    id: int
    email: EmailStr
