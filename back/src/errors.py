from fastapi import HTTPException, status


def with_errors(*errors: HTTPException):
    d = {}
    for err in errors:
        if err.status_code in d:
            d[err.status_code]["description"] += f"\n\n{err.detail}"
        else:
            d[err.status_code] = {"description": err.detail}
    return d


def learning_session_not_found():
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Learning session not found")


def RTSP_not_found():
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Not Found')


def invalid_credentials():
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")


def unauthorized():
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Need authorization")


def token_expired():
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")


def token_validation_failed():
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad token specified")


def unable_to_create_account():
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to sign up with such credentials")
