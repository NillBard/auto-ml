import os
import os
import time
import zipfile
import logging

import pandas as pd
from fastapi import Depends
from ultralytics import YOLO
from tempfile import TemporaryDirectory
from cvat_module.new_cvat import cvat_export_dataset
from schemas.train import TrainingConf
from celery import Celery, chord, chain, group
from pathlib import Path

import os.path as p

from db import Session
from models.user import TrainProject, TrainingConfiguration
from s3.s3 import s3
from settings import settings
from db.session import _session
from db import get_database
from sqlalchemy.orm import Session
from .yolo_trainer.train import train_yolo
from .ignite_trainer.train import train_task_model

import numpy as np

def is_image(filename):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    return any(filename.lower().endswith(ext) for ext in image_extensions)

celery = Celery(__name__)
celery.conf.broker_url = settings.CELERY_BROKER_URL
celery.conf.result_backend = settings.CELERY_RESULT_BACKEND

@celery.task(name="chose_best")
def choose_best(models_metrics): 
  print('----------choose---------')
  print(models_metrics)
  print('----------choose---------')

  best_model = None
  best_avg_metric = -np.inf
    
  for model in models_metrics:
      total_score = model['recall'] + model['precision'] - model['loss']

      if total_score > best_avg_metric:
          best_avg_metric = total_score
          best_model = model['id']
          
  return best_model

@celery.task(name="finish_train")
def select_project_best(conf_id: int, project_id: int): 
    print('----------finish_train---------')
    print(f'select: {conf_id} in {project_id}')
    print('----------finish_train---------')
    db = _session()
    project = db.query(TrainProject).filter_by(id=project_id).first()

    project.best_model_id=conf_id
    project.status = 'processed'
    db.commit()

# @celery.task(name="test_item")
# def process_item(item): 
#   print(item)
#   time.sleep(5)
#   return item

# @celery.task(name="test")

# def test_parallel():
#     models_metrics = [
#        {
#         'id': 'Model_A',
#         'precision': 0.80,
#         'recall': 0.85,
#         'loss': 0.15,
#       },  # [Accuracy, F1-Score, ROC AUC]
#       {
#         'id': 'Model_B',
#         'precision': 0.85,
#         'recall': 0.80,
#         'loss': 0.12,
#       },      
#       {
#         'id': 'Model_C',
#         'precision': 0.83,
#         'recall': 0.83,
#         'loss': 0.13,
#       },
#     ]
#     print('--------')
#     print('start')
#     res = chord(process_item.s(item) for item in models_metrics)(process_item_callback.s())
#     # model = res.get()
#     print('--------')
#     print(f'model: {model}')
#     print('--------')

@celery.task(name="train_models")
def train_models(project_id:int, dataset_id: int, cvat_token:str):
  db = _session()
  configs = db.query(TrainingConfiguration).filter_by(training_project_id=project_id).all()
  print('--------')
  print(f'project_id: {project_id}')
  [print(conf.model) for conf in configs]
  # print(print(conf) for conf in configs)
  print('--------')
  group_task = [
     chain(
        prepare_training_data.s(
          conf_id=conf.id, 
          dataset_id=dataset_id, 
          cvat_token=cvat_token,
          task_index = i
        ),
        train.s()
      )
      for i, conf in enumerate(configs)
    ]
  chord(
    group(group_task)
  )(
    chain(
      choose_best.s(),
      select_project_best.s(project_id)
    )
  )

  # return ''


@celery.task(name="prepare_train")
def prepare_training_data(
                          conf_id: int,
                          dataset_id: int,
                          cvat_token: str,
                          task_index: int
                        ):
    db = _session()
    conf = db.query(TrainingConfiguration).filter_by(id=conf_id).first()
    print(conf.training_conf)
    if 'yolo' in conf.model.lower():
      format = 'YOLO 1.1'
    elif conf.training_conf['task_type'] == 'classification':
      format = 'ImageNet 1.0'
    else: 
      format = 'COCO 1.0'

    time.sleep(task_index*5)
    export_status = cvat_export_dataset(cvat_token, dataset_id, conf.id, format)

    if export_status == 202:
        conf.dataset_s3_location = f"/{conf.id}-dataset-{dataset_id}.zip"
        db.commit()
        bucket = f"cvat"
        while not s3.has_file(conf.dataset_s3_location, bucket):
          time.sleep(5)
          return conf_id

    # elif export_status == 409:
    #     while True:
    #       export_status = cvat_export_dataset(cvat_token, dataset_id, conf.id, format)
           
    #     return conf_id
    else:
      raise 'error'
    
@celery.task(name="train")
def train(conf_id: int):
    db = _session()
    conf = db.query(TrainingConfiguration).filter_by(id=conf_id).first()
    print('---')
    print(conf)
    print('---')
    # yolo = YOLO(str(conf.model) + '.yaml')
    # conf.status = 'processing'
    # bucket=f"cvat"
    # train_conf = conf.training_conf
    if 'yolo' in conf.model.lower():
      res = train_yolo(conf_id)
    else:
      res = train_task_model(conf_id)
    return res



# def train(conf_id: int, user_id: int):
#     db = _session()
#     conf = db.query(TrainingConfiguration).filter_by(id=conf_id, created_by=user_id).first()
#     yolo = YOLO(str(conf.model) + '.yaml')
#     conf.status = 'processing'
#     bucket=f"cvat"

#     db.commit()
#     try:
#         with TemporaryDirectory(dir=os.getcwd()) as tmp:


#             logging.info("starting download dataset")
#             with open(tmp + conf.dataset_s3_location, mode='w+b') as f:
#                 print(conf.dataset_s3_location),
#                 print(f"user-{user_id}")
#                 # bucket=f"user-{user_id}"
#                 bucket=f"cvat"
#                 print(f'bucket: {bucket}')
#                 s3.download_file(f, conf.dataset_s3_location, bucket)
            
            
#             logging.info("starting extract data")
#             with zipfile.ZipFile(tmp + conf.dataset_s3_location, 'r') as f:
#                 print(f.namelist())
#                 f.extract('obj.names', path=tmp + '/data')
#                 [f.extract(file, path=tmp + '/data/images/') for file in f.namelist() if (is_image(file)) and file.startswith('obj_train_data')]
#                 [f.extract(file, path=tmp + '/data/images/') for file in f.namelist() if (is_image(file) and file.startswith('obj_validation_data'))]
#                 [f.extract(file, path=tmp + '/data/labels/') for file in f.namelist() if (file.endswith('.txt') and file.startswith('obj_train_data'))]
#                 [f.extract(file, path=tmp + '/data/labels/') for file in f.namelist() if (file.endswith('.txt') and file.startswith('obj_validation_data'))]

#             path = Path(tmp + '/data/images/obj_train_data')
#             path.rename(tmp + '/data/images/train')

#             path = Path(tmp + '/data/labels/obj_train_data')
#             path.rename(tmp + '/data/labels/train')

#             print(os.listdir(tmp + '/data/images/train/'))
#             logging.info("generating yaml file")
#             classes = []
#             with open(tmp + '/data/obj.names', 'r') as f:
#                 for line in f.readlines():
#                     classes.append(line.replace('\n', ''))
#             print(classes)
            
            
#             training_conf = conf.training_conf
#             training_conf['classes'] = classes
#             conf.training_conf = training_conf
#             db.commit()
            
            
#             with open(tmp + '/data/data.yaml', 'w') as f:
#                 names = [f'  {classes.index(name)}: {name}\n' for name in classes]
#                 f.writelines(['path: ' + tmp + '/data/\n',
#                             'train: images/train\n',
#                             'val: images/train\n',
#                             'names:\n'] + names)

#             yolo.train(
#                 data=tmp + '/data/data.yaml',
#                 project=tmp,
#                 name=conf.name,
#                 epochs=conf.training_conf['epochs'],
#                 # patience=conf.training_conf['patience'],
#                 batch=conf.training_conf['batch'],
#                 # imgsz=conf.training_conf['imgsz'],
#                 optimizer=conf.training_conf['optimizer'],
#                 device=conf.training_conf['device'] if conf.training_conf['device'] == 'cpu' else 0
#             )

#             yolo.export(format='onnx')

#             data = pd.read_csv(tmp + f'/{conf.name}/results.csv')
#             data = data.to_dict()
#             clean_data = {}

#             for key in data.keys():
#                 if key.strip() == 'metrics/precision(B)':
#                     clean_data['precision'] = data[key]
#                 elif key.strip() == 'metrics/recall(B)':
#                     clean_data['recall'] = data[key]
#                 elif key.strip() == 'train/box_loss':
#                     clean_data['train/box_loss'] = data[key]
#                 elif key.strip() == 'train/cls_loss':
#                     clean_data['train/cls_loss'] = data[key]
#                 elif key.strip() == 'train/dfl_loss':
#                     clean_data['train/dfl_loss'] = data[key]
#                 elif key.strip() == 'metrics/mAP50(B)':
#                     clean_data['mAP50'] = data[key]
#                 elif key.strip() == 'metrics/mAP50-95(B)':
#                     clean_data['mAP50-95'] = data[key]
#                 elif key.strip() == 'val/box_loss':
#                     clean_data['val/box_loss'] = data[key]
#                 elif key.strip() == 'val/cls_loss':
#                     clean_data['val/cls_loss'] = data[key]
#                 elif key.strip() == 'val/dfl_loss':
#                     clean_data['val/dfl_loss'] = data[key]

#             conf.result_metrics = clean_data
#             db.commit()

#             with open(tmp + f'/{conf.name}/weights/best.pt', 'rb') as f:
#                 path = f'/user/{conf_id}/result/best.pt'
#                 s3.upload_file(f, path, bucket)

#             with open(tmp + f'/{conf.name}/weights/best.onnx', 'rb') as f:
#                 path = f'/user/{conf_id}/result/best.onnx'
#                 s3.upload_file(f, path, bucket)

#         conf.weight_s3_location = f'/user/{conf_id}/result/best.pt'
#         conf.onnx_s3_location = f'/user/{conf_id}/result/best.onnx'

#         conf.status = 'processed'
#         db.commit()
#         print('hehe')
#     except Exception as e:
#         print(e)
#         conf.status = 'error'
#         db.commit()

from models.processing import Processing
from inference.inference_service import InferenceService

@celery.task(name="inference")
def inference(pipeline_id: int):
    """
    Таск для выполнения инференса.
    
    Args:
        pipeline_id: Идентификатор конфигурации.
    """
    db = _session()
    pipeline = db.query(Processing).filter_by(id=pipeline_id).first()
    pipeline.status = 'processing' 
    db.commit()

    try:
        # Создаем экземпляр сервиса
        service = InferenceService(
            rtsp_url=pipeline.rtsp_url,
            trigger_class=pipeline.trigger_class,  # класс, который мы ищем
            bucket="inference",
            prefix=f'{pipeline.id}',
            is_custom=pipeline.is_custom,
            model_path=pipeline.model,
            confidence_threshold=pipeline.confidence_threshold
        )
        service.run()

        pipeline.status = 'processed'
        db.commit()

    except Exception as e:
        print(e)
        pipeline.status = 'error'
        db.commit()
