import requests
import json

from settings import settings

# CVAT_URL = 'http://localhost:8081/'

def cvat_register_user(
        login: str,
        password: str,
        email: str
):
    data = json.dumps({
        "username": login,
        "email": email,
        "password1": password,
        "password2": password,
    })
    header = {"content-type": "application/json"}
    response = requests.post(settings.CVAT_URL + "api/auth/register", headers=header, data=data)
    if response.status_code == 200:
        print('exellent')
    else:
        print(response.text)


def cvat_login_user(
        password: str,
        email: str,
):
    session = requests.Session()
    # header = {"content-type": "application/json"}
    data = json.dumps({
        "username": email,
        "password": password,
    })
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
        return response
    else:
        print(response.content)


def cvat_get_projects(sessionid, csrftoken):
    header = {
        "content-type": "application/json"
    }
    cookie = {
        'sessionid': sessionid,
        'csrftoken': csrftoken
    }
    response = requests.get(settings.CVAT_URL + "api/projects",headers=header, cookies=cookie)
    if response.status_code == 200:
        return response.json()
    else:
        print(response.text)


def cvat_s3_create(sessionid: str, csrftoken: str, user_id: int):
    header = {"content-type": "application/json"}
    cookie = {
        'sessionid': sessionid,
        'csrftoken': csrftoken
    }

    header['x-csrftoken'] = csrftoken
    data = json.dumps({
        "display_name": "s3",
        "credentials_type": "KEY_SECRET_KEY_PAIR",
        "provider_type": "AWS_S3_BUCKET",
        "resource": f"user-{user_id}",
        "key": settings.AWS_ACCESS_KEY_ID,
        "secret_key": settings.AWS_SECRET_ACCESS_KEY,
        "specific_attributes": f"endpoint_url={settings.AWS_HOST}"
    })

    response = requests.post(settings.CVAT_URL + "api/cloudstorages", headers=header, data=data, cookies=cookie)
    
    if response.status_code == 201:
        print('true')
    else:
        print(response.content)

def cvat_export_dataset(sessionid: str, csrftoken: str, dataset_id: int, conf_id: int):
    header = {"content-type": "application/json"}
    cookie = {
        'sessionid': sessionid,
        'csrftoken': csrftoken
    }

    header['x-csrftoken'] = csrftoken


    response = requests.get(settings.CVAT_URL + "api/cloudstorages", headers=header, cookies=cookie)

    data = {
        "action": "download",
        "cloud_storage_id": response.json().get("results")[0].get("id"),
        "filename": f"{conf_id}-dataset-{dataset_id}.zip",
        "format": "YOLO 1.1",
        "location": "cloud_storage"
    }

    response = requests.get(settings.CVAT_URL + f"api/projects/{dataset_id}/dataset/", headers=header, params=data, cookies=cookie)
    response = requests.get(settings.CVAT_URL + f"api/projects/{dataset_id}/dataset/", headers=header, params=data, cookies=cookie)
    print(response.status_code)
    print(response.content)
    return response.status_code


def cvat_delete_project(
        sessionid: str, 
        csrftoken: str, 
        dataset_id: int
):
    header = {"content-type": "application/json"}
    cookie = {
        'sessionid': sessionid,
        'csrftoken': csrftoken
    }

    header['x-csrftoken'] = csrftoken
    response = requests.delete(settings.CVAT_URL + f'api/projects/{dataset_id}', headers=header, cookies=cookie)

    return response.status_code

    

if __name__ == "__main__":
    cvat_login_user("qwertyqwe", "user@ma.com", 1)
