from typing import Optional

from pydantic import BaseSettings
# from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    JWT_SECRET: str = 'super_puper_long_secure_secret' # sign key
    JWT_ACCESS_EXPIRE: int = 5  # In minutes
    JWT_REFRESH_EXPIRE: int = 60  # In minutes
    JWT_REFRESH_LONG_EXPIRE: int = 12  # In hours

    DB_ADDR: str = "172.16.1.10"
    DB_PORT: int = 15432
    DB_USERNAME: str = "postgres"
    DB_PASSWORD: str = 'password'
    DB_NAME: str = 'automl'

    AWS_HOST: str = 'http://172.16.1.10:19000'
    AWS_ACCESS_KEY_ID: str = 'automl'
    AWS_SECRET_ACCESS_KEY: str = 'ViVyOUQtHVfGaZuOVm1yhro5YLFJcsYf'
    AWS_REGION: Optional[str] = None
    AWS_BUCKET: str = 'cvat'

    CELERY_BROKER_URL: str = 'redis://172.16.1.10:16379/0'
    CELERY_RESULT_BACKEND: str = 'redis://172.16.1.10:16379/0'
    KAFKA_URL: str = "kafka:9092"  # Используем внутренний порт Kafka

    SERVER_ADDR: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    SERVER_TEST: bool = True

    CVAT_URL:str = 'http://172.16.0.190:8080/'
    CVAT_URL_HOST:str = 'http://172.16.0.190:8080/'
    class Config:
        env_file = "../.env"


settings = Settings()
