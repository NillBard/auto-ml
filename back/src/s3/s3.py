import boto3
import botocore

import logging
import os

import botocore.exceptions

from settings import settings


class S3:
    def __init__(self):
        self.session = boto3.session.Session(aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                             aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                             region_name=settings.AWS_REGION)
        self.s3client = self.session.client(service_name='s3', endpoint_url=settings.AWS_HOST)
        # self.create_bucket(settings.AWS_BUCKET)

    def has_file(self, fileid: str, bucket: str):
        try:
            self.s3client.head_object(Bucket=bucket, Key=fileid)
            return True
        except botocore.exceptions.ClientError:
            return False

    def upload_file(self, file, fileid: str, bucket: str):
        self.s3client.upload_fileobj(file, bucket, fileid)

    def download_file(self, file, fileid: str, bucket: str):
        try:
            print(f"{file} {fileid} {bucket}")
            self.s3client.head_object(Bucket=bucket, Key=fileid)
            self.s3client.download_fileobj(bucket, fileid, file)
            file.seek(0)
        except botocore.exceptions.ClientError:
            raise FileNotFoundError("File not found")

    def delete_file(self, fileid: str, bucket: str):
        self.s3client.delete_object(Bucket=bucket, Key=fileid)

    def create_bucket(self, name):
        try:
            try:
                self.s3client.head_bucket(Bucket=name)
                return
            except botocore.exceptions.ClientError:
                pass
            self.s3client.create_bucket(Bucket=name)
        except botocore.exceptions.ClientError as e:
            raise e
        except Exception as ex:
            logging.info(ex)

    def generate_link(self, bucket, key):
        return self.s3client.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=3600
        )

    def list_buckets(self):
        """Возвращает список всех бакетов в S3"""
        try:
            print(self.s3client)
            response = self.s3client.list_buckets()
            return [bucket['Name'] for bucket in response['Buckets']]
        except botocore.exceptions.ClientError as e:
            raise e

    def download_files_with_prefix(self, bucket: str, prefix: str, local_dir: str) -> list:
        """Скачивает все файлы из бакета с указанным префиксом в локальную директорию."""
        try:
            # Получаем список объектов с указанным префиксом
            print('Получаем список объектов с указанным префиксом')
            print(f'{bucket} {prefix}')
            response = self.s3client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            downloaded_files = []

            if 'Contents' in response:
                for obj in response['Contents']:
                    file_key = obj['Key']
                    print(file_key)
                    local_file_path = os.path.join(local_dir, os.path.basename(file_key))
                    with open(local_file_path, 'wb') as file:
                        self.download_file(file, file_key, bucket)
                        downloaded_files.append(local_file_path)

            return downloaded_files
        except botocore.exceptions.ClientError as e:
            print(e)
            raise e

s3 = S3()

s3.create_bucket('inference')