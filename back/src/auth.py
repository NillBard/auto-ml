import re
import json
import jwt
import uuid
from typing import Dict, Any, Type, Optional
from datetime import datetime, timezone, timedelta

from fastapi import Response, Cookie, Depends, Request

import errors
from db import Session, get_database
from settings import settings
from models.user import User
from schemas.login import Refresh

path = '/'


def set_cookie(access: str, response: Response, max_age: int):
    response.set_cookie('access', access, httponly=True, samesite='lax', max_age=max_age, path=path)


def encode_token(payload) -> str:
    return jwt.encode(payload, settings.JWT_SECRET, algorithm='HS256')


def decode_token(token: str, token_type: str, suppress: bool = False) -> Dict[str, Any]:
    try:
        data = jwt.decode(token, settings.JWT_SECRET, algorithms=['HS256'],
                          options={"require": ["exp", "role"]})
        if data["role"] != token_type:
            raise errors.token_validation_failed()
        return data
    except jwt.ExpiredSignatureError:
        if suppress:
            data = jwt.decode(token, settings.JWT_SECRET, algorithms=['HS256'],
                              options={"verify_signature": False})
            if data["role"] != token_type:
                raise errors.token_validation_failed()
            return data
        raise errors.token_expired()
    except jwt.DecodeError:
        raise errors.token_validation_failed()


def init_tokens(account: Type[User], response: Response):
    now = datetime.now(timezone.utc)
    access_payload = {
        "role": "access",
        'user_id': account.id,
        "exp": now + timedelta(minutes=settings.JWT_ACCESS_EXPIRE)
    }
    refresh_payload = {
        "role": "refresh",
        "user_id": account.id,
        "exp": now + timedelta(hours=settings.JWT_REFRESH_EXPIRE)
    }

    access = encode_token(access_payload)
    refresh = encode_token(refresh_payload)

    print(access)
    set_cookie(access, response, settings.JWT_REFRESH_EXPIRE * 3600)
    print(Cookie(None))
    return Refresh(refresh=refresh)


def verify_access(access: Optional[str], db: Session) -> Type[User]:
    if access is None:
        raise errors.unauthorized()
    access_payload = decode_token(access, 'access')
    user = db.query(User).filter_by(id=access_payload['user_id']).first()
    return user


def refresh_tokens(access: Optional[str], refresh: str, response: Response, db: Session):
    if access is None:
        raise errors.unauthorized()
    access_payload = decode_token(access, 'access', suppress=True)
    refresh_payload = decode_token(refresh, 'refresh')

    if access_payload['user_id'] != refresh_payload['user_id']:
        raise errors.token_validation_failed()

    user = db.query(User).filter_by(id=access_payload['user_id']).first()
    print(user, flush=True)
    now = datetime.now(timezone.utc)
    access_payload = {
        "role": "access",
        'user_id': user.id,
        "exp": now + timedelta(minutes=settings.JWT_ACCESS_EXPIRE)
    }
    refresh_payload = {
        "role": "refresh",
        "user_id": user.id,
        "exp": now + timedelta(hours=settings.JWT_REFRESH_EXPIRE)
    }

    access = encode_token(access_payload)
    refresh = encode_token(refresh_payload)

    set_cookie(access, response, settings.JWT_REFRESH_EXPIRE * 3600)
    return Refresh(refresh=refresh)


async def get_user(access: Optional[str] = Cookie(None),
                   db: Session = Depends(get_database)):
    print(access)
    user = verify_access(access, db)
    return user
