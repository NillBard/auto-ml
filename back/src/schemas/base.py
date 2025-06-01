import json
from pydantic import BaseModel as PydanticBaseModel


def custom_json_dumps(obj, **kwargs):
    return json.dumps(obj, **kwargs, ensure_ascii=False, allow_nan=False, indent=None, separators=(',', ':'))


class BaseModel(PydanticBaseModel):
    class Config:
        orm_mode = True
        json_dumps = custom_json_dumps
