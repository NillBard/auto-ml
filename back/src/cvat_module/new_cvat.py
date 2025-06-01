import time
from fastapi import Cookie
import requests
import json

from settings import settings
from urllib.parse import quote, urlencode

def cvat_login_user(
        password: str,
        email: str,
):
    session = requests.Session()
    # data = json.dumps({
    #     "username": email,
    #     "password": password,
    # })
    print(f'cvat_login_user: {settings.CVAT_URL_HOST}')
    auth_url = f"{settings.CVAT_URL_HOST}/api/auth/login"
    # print(f'host: {auth_url}')

    response = session.post(
        auth_url, 
        json={
          "username": email.split('@')[0],
          "password": password,
        }
    # data=data
    )
    print(f'status: {response.status_code}')
    if response.status_code == 200:
        return response.json().get('key')
    else:
        print(response.content)

def cvat_s3_create(cvat_token:str ):
    header = {
      "content-type": "application/json",
      "Authorization": f"Token {cvat_token}",
    }
    # cookie = {
    #     'sessionid': sessionid,
    #     'csrftoken': csrftoken
    # }

    # header['x-csrftoken'] = csrftoken
    data = json.dumps({
        "display_name": "s3",
        "credentials_type": "KEY_SECRET_KEY_PAIR",
        "provider_type": "AWS_S3_BUCKET",
        "resource": f"cvat",
        "key": settings.AWS_ACCESS_KEY_ID,
        "secret_key": settings.AWS_SECRET_ACCESS_KEY,
        "specific_attributes": f"endpoint_url={settings.AWS_HOST}"
    })

    response = requests.get(settings.CVAT_URL_HOST + "api/cloudstorages", headers=header, data=data)
    print('s3--------------------------')
    print('s3')
    print('s3')
    if response.status_code == 201:
        print('true')
    else:
        print(response.content)

def cvat_get_tasks(cvat_token: str):
    header = {
      "Authorization": f"Token {cvat_token}",
      "content-type": "application/json"
    }
    # cookie = {
    #     'sessionid': sessionid,
    #     'csrftoken': csrftoken
    # }

    url = f'{settings.CVAT_URL_HOST}/api/tasks'
    response = requests.get(
      url,
      headers=header, 
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(response.text)

def custom_encoder(params):
  return '&'.join(
      f"{k}={quote(str(v), safe='+')}" for k, v in params.items()
  )

def cvat_export_dataset(cvat_token: str, dataset_id: int, conf_id: int, format: str):
  try:
    print('---------------')
    print(cvat_token)
    print('---------------')
    header = {
      "Authorization": f"Token {cvat_token}",
      "content-type": "application/json"
    }
    response = requests.get(settings.CVAT_URL_HOST + "api/cloudstorages", headers=header)
    print('---------------')
  
    print(response.json().get("results")[0].get("id"))
  
    print('---------------')
  

    print(format)
    params = {
        "cloud_storage_id": response.json().get("results")[0].get("id"),
        "filename": f"{conf_id}-dataset-{dataset_id}.zip",
        "format": format,
        "location": "cloud_storage",
        'save_images': True,
    }
    
    export_url = f"{settings.CVAT_URL_HOST}api/tasks/{dataset_id}/dataset/export?{custom_encoder(params)}"
    print(export_url)

    response = requests.post(export_url, headers=header)

    # response = requests.get(settings.CVAT_URL + f"api/projects/{dataset_id}/dataset/", headers=header, params=data, cookies=cookie)
    print(response.status_code)
    print(response.content)
    return response.status_code
  except Exception as e:
      print(e)
