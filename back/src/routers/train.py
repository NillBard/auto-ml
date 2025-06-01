import os.path as p
import time
from typing import List

from fastapi import APIRouter, Depends, Cookie, status, Response
from fastapi.responses import RedirectResponse

from starlette.responses import JSONResponse

from cvat_module.new_cvat import cvat_export_dataset
from db import get_database, Session
from schemas.train import TrainingConf, TrainingConfGetFull, TrainingProjectFull
from s3.s3 import s3
# from cvat_module.cvat import cvat_export_dataset
from models.user import TrainProject, TrainingConfiguration
from mlcore.celery_app import train, train_models
from auth import get_user
import errors

def get_models_by_type(task_type:str):
  if task_type == 'classification':
    return [
      {'label':'resnset', 'value': 'resnet50'},
      {'label':'efficientnet', 'value': 'efficientnet'},
      {'label':'mobilenet', 'value': 'mobilenet'}
    ]
  elif task_type == 'detection':
    return [
      { 'label': 'ssd', 'value': 'ssd'}, 
      { 'label': 'fasterrcnn', 'value': 'fasterrcnn'}, 
      { 'label': "yolov8n", 'value': "yolov8n" },
      { 'label': "yolov8s", 'value': "yolov8s" },
      { 'label': "yolov8m", 'value': "yolov8m" },
      { 'label': "yolov8l", 'value': "yolov8l" },
      { 'label': "yolov8x", 'value': "yolov8x" },
    ]
    
  elif task_type == 'segmentation':
    return [
       { 'label': 'deeplabv3', 'value': 'deeplabv3'}, 
       { 'label': 'fcn', 'value': 'fcn'}, 
       { 'label': 'unet', 'value': 'unet'},
      ]

def get_conf_without_model(conf: TrainingConf):
    return conf.model_dump(exclude={'models'})


router = APIRouter()

@router.get('/all', response_model=List[TrainingProjectFull])
async def get_all_configurations(db: Session = Depends(get_database),
                                 user=Depends(get_user)
                                 ):
    if user is None:
        raise errors.unauthorized()
    projects = db.query(TrainProject).all()
    print('-------')
    print(projects[0].id)
    print('-------')
    return projects

@router.get('/all-conf', response_model=List[TrainingConfGetFull])
async def get_all_trained_configurations(db: Session = Depends(get_database), user=Depends(get_user)):
    if user is None:
        raise errors.unauthorized()
    conf = db.query(TrainingConfiguration).all()
    return conf

@router.get('/all-trained', response_model=List[TrainingConfGetFull])
async def get_all_trained_configurations(db: Session = Depends(get_database), user=Depends(get_user)):
    if user is None:
        raise errors.unauthorized()
    conf = db.query(TrainingConfiguration).filter(
        TrainingConfiguration.status == 'processed',
        TrainingConfiguration.dataset_s3_location.isnot(None),
        TrainingConfiguration.weight_s3_location.isnot(None)
    ).all()
    return conf

@router.get('/{conf_id}', response_model=TrainingConfGetFull)
async def get_conf_by_id(conf_id: int,
                         db: Session = Depends(get_database),
                         user=Depends(get_user)
                         ):
    if user is None:
        raise errors.unauthorized()
    conf = db.query(TrainingConfiguration).filter_by(id=conf_id).first()
    return conf

@router.get('/project/{conf_id}', response_model=TrainingProjectFull)
async def get_conf_by_id(conf_id: int,
                         db: Session = Depends(get_database),
                         user=Depends(get_user)
                         ):
    if user is None:
        raise errors.unauthorized()
    conf = db.query(TrainProject).filter_by(id=conf_id).first()
    trains = db.query(TrainingConfiguration).filter_by(training_project_id=conf_id).all()
    conf.trains = trains
    return conf

@router.post('/')
async def create_traininge_project(params: TrainingConf,
                               db: Session = Depends(get_database),
                               user=Depends(get_user),
                               cvat_token: str = Cookie(None)):
    if user is None:
        raise errors.unauthorized()
    print(params)

    # if (params.models == 'auto'):
    #   models = get_models_by_type(params.task_type)
    # else:
    #   models = params.models
    models = params.models
    new_project = TrainProject(
        name=params.name,
        created_by = user.id
    )
    db.add(new_project)
    db.commit()

    project_configs = [create_configuration(model, new_project.id, params) for model in models]
    db.add_all(project_configs)
    db.commit()

    task = train_models.delay(
      project_id=new_project.id, 
      dataset_id=params.dataset_id, 
      cvat_token=cvat_token
    )

    return JSONResponse({"task_id": task.id})



def create_configuration( model: str,
                          project_id: int,
                          params: TrainingConf
                              #  db: Session = Depends(get_database),
                              #  user=Depends(get_user),
                              #  cvat_token: str = Cookie(None)
                          ):
    # if user is None:
    #     raise errors.unauthorized()
    print(params)
    training_params = {
        'epochs': params.epochs,
        'time': params.time,
        'patience': params.patience,
        'batch': params.batch,
        'imgsz': params.imgsz,
        'optimizer': params.optimizer,
        'classes': params.class_names,
        'device': params.device,
        'dataset_id': params.dataset_id,
        'task_type': params.task_type
    }
    new_conf = TrainingConfiguration(
        model=model,
        training_conf=training_params,
        training_project_id=project_id,
    )  
    return new_conf

# @router.post('/')
# async def create_configuration( model: str,
#                                 params: TrainingConf,
#                                 project_id: int,
                              #  db: Session = Depends(get_database),
                              #  user=Depends(get_user),
                              #  cvat_token: str = Cookie(None)
                              #  ):
    # if user is None:
    #     raise errors.unauthorized()
    # print(params)
    # training_params = {
    #     'epochs': params.epochs,
    #     'time': params.time,
    #     'patience': params.patience,
    #     'batch': params.batch,
    #     'imgsz': params.imgsz,
    #     'optimizer': params.optimizer,
    #     'classes': params.class_names,
    #     'device': params.device,
    #     'dataset_id': params.dataset_id,
    #     'task_type': params.task_type
    # }

    # new_conf = TrainingConfiguration(
    #     model=params.model,
    #     training_conf=training_params,
    #     created_by = user.id
    # )
    # return new_conf
    # db.add(new_conf)
    # db.commit()

    # if 'yolo' in params.model.lower():
    #   format = 'YOLO 1.1'
    # elif params.task_type == 'classification':
    #   format = 'ImageNet 1.0'
    # else: 
    #   format = 'COCO 1.0'

    # export_status = cvat_export_dataset(cvat_token, params.dataset_id, new_conf.id, format)
    # print('-----------')
    # print('here', export_status)
    # print('task_type', params.task_type)
    # print('-----------')
    # if export_status == 202:
    #     new_conf.dataset_s3_location = f"/{new_conf.id}-dataset-{params.dataset_id}.zip"
    #     db.commit()
    #     bucket = f"cvat"
    #     while not s3.has_file(new_conf.dataset_s3_location, bucket):
    #       time.sleep(5)

    #     task = train.delay(new_conf.id, user.id)

    #     return JSONResponse({"task_id": task.id})
    # else:
    #     return Response(status_code=status.HTTP_400_BAD_REQUEST)
    
@router.get('/models/{task_type}')
async def get_file(task_type: str):
  if task_type in ['classification', 'detection', 'segmentation']:
    models = get_models_by_type(task_type)
    return JSONResponse(models)

  else:
    return Response(status_code=status.HTTP_400_BAD_REQUEST)
     
        


@router.delete('/{conf_id}')
async def delete_conf(conf_id: int,
                      db: Session = Depends(get_database),
                      user = Depends(get_user)):
    if user is None:
        raise errors.unauthorized()
    conf = db.query(TrainingConfiguration).filter_by(id=conf_id).first()
    if conf is None:
        raise
    db.delete(conf)


@router.get('/{conf_id}/{file_type}', status_code=302, response_class=RedirectResponse)
async def get_file(conf_id: int,
                   file_type: str,
                   db: Session = Depends(get_database),
                   user = Depends(get_user)):
    if user is None:
        raise errors.unauthorized()
    conf = db.query(TrainingConfiguration).filter_by(id=conf_id).first()
    if conf is None:
        raise
    if file_type == 'dataset':
        if conf.dataset_s3_location is None:
            raise
        return conf.s3_dataset_url
    elif file_type == 'pt':
        if conf.weight_s3_location is None:
            raise
        return conf.s3_weight_url
    elif file_type == 'onnx':
        return conf.s3_onnx_url
    else:
        raise


  
  
  
  
  
  
  
    # task = train.delay(new_conf.id, user.id)

    # return JSONResponse({"task_id": task.id})


# @router.post('/{conf_id}/dataset', status_code=204)
# async def upload_dataset(conf_id: int,
#                          dataset: UploadFile = File(...),
#                          db: Session = Depends(get_database),
#                          user=Depends(get_user)
#                          ):
#     if user is None:
#         raise errors.unauthorized()
#     data = await dataset.read()
#     try:
#         training = db.query(TrainingConfiguration).filter_by(id=conf_id).first()
#         if training.dataset_s3_location is not None:
#             raise
#         with TemporaryDirectory(prefix='data') as tmp:
#             with open(p.join(tmp, dataset.filename), 'wb') as f:
#                 f.write(data)
#             with open(p.join(tmp, dataset.filename), 'rb') as f:
#                 path = f'/user/{conf_id}/dataset/{dataset.filename}'
#                 s3.upload_file(f, path)
#         training.dataset_s3_location = path
#         db.commit()
#     except Exception:
#         print('error')
#         raise


# @router.post('/{conf_id}/start')
# async def start_training(conf_id: int,
#                          db: Session = Depends(get_database),
#                          user=Depends(get_user)
#                          ):
#     if user is None:
#         raise errors.unauthorized()
#     conf = db.query(TrainingConfiguration).filter_by(id=conf_id).first()
#     if conf is None:
#         raise
    
#     return JSONResponse({"task_id": task.id})