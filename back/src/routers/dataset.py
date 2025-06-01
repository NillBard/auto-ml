from fastapi import APIRouter, Cookie, Depends, status, HTTPException
from requests import Session

# from cvat.cvat import cvat_get_projects, cvat_delete_project
from auth import get_user
from cvat_module.new_cvat import cvat_get_tasks
from models.user import User
from db import get_database
from cvat_module.cvat import cvat_get_projects, cvat_delete_project
from schemas.dataset import ProjectsSchema

router = APIRouter()

@router.get("")
async def get_all(cvat_token: str = Cookie(None), csrftoken: str = Cookie(None), user=Depends(get_user),  db: Session= Depends(get_database)):
    # projects = cvat_get_projects(sessionid, csrftoken)
    tasks = cvat_get_tasks(cvat_token)
    print(f"{user.id}", flush=True)
    print('here', flush=True)
    user = db.query(User).filter_by(id=user.id).first()
    print(f'name: {user.email}')
    print(f'pass: {user.password}')
    response = []
    for task in tasks.get("results"):
        response.append(
            ProjectsSchema(
                id=task.get("id"),
                name=task.get("name"),
                created_date=task.get("created_date"),
                status=task.get("status")
            )
        )
    return response


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: int, 
    sessionid: str = Cookie(None), 
    csrftoken: str = Cookie(None)
):
    code = cvat_delete_project(dataset_id=dataset_id,
                        sessionid=sessionid,
                        csrftoken=csrftoken)
    if code != 204:
        raise HTTPException(status_code=code, detail='error')