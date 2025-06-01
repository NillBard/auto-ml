import logging
import os
from tempfile import TemporaryDirectory
import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any
from ignite.engine import Engine
import zipfile

from models.user import TrainingConfiguration
from .detection import get_num_classes_from_annotations, train_model as train_detection_model
from .classification import train_model as train_classification_model
from .segmentation import train_segmentation_model 
from .utils import Config, TrainingHistory, convert_numpy_types, prepare_classification_dataset, prepare_detection_dataset, prepare_segmentation_dataloader, save_model
from db.session import _session
from celery import Celery
from settings import settings
from s3.s3 import s3

class BaseConfig:
    LR = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    VAL_SPLIT = 0.2  # Доля данных для валидации
    NUM_WORKERS = 4  # Количество воркеров для загрузки данных

def unzip_dataset(tmp, location, task_type):
  print('-----------')
  print(task_type)
  print(location)
  print('-----------')

  if task_type == 'segmentation':
    with zipfile.ZipFile(tmp + location, 'r') as f:
      f.extractall(path=tmp + '/data')
    data_root=tmp + '/data/images/default'
    ann_file=tmp + '/data/annotations/instances_default.json'
    return data_root, ann_file
  elif task_type  == 'detection':
    with zipfile.ZipFile(tmp + location, 'r') as f:
      f.extractall(path=tmp + '/data')
    data_root=tmp + '/data/images/default'
    ann_file=tmp + '/data/annotations/instances_default.json'
    return data_root, ann_file
  elif task_type  == 'classification':
    with zipfile.ZipFile(tmp + location, 'r') as f:
      f.extractall(path=tmp + '/data')
    data_root=tmp + '/data/'
    ann_file=''
    return data_root, ann_file

  else:
      raise ValueError(f"Unknown task type: {task_type}")

def prepare_dataloader(
    task_type: str,
    data_root: str,
    ann_file: Optional[str] = None,
    batch_size: int = Config.BATCH_SIZE
) -> DataLoader:

    if task_type == 'segmentation':
        return prepare_segmentation_dataloader(data_root, ann_file, batch_size)
    elif task_type == 'detection':
        return prepare_detection_dataset(data_root, ann_file, batch_size)
    elif task_type == 'classification':
        return prepare_classification_dataset(data_root, batch_size)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

# celery = Celery(__name__)
# celery.conf.broker_url = settings.CELERY_BROKER_URL
# celery.conf.result_backend = settings.CELERY_RESULT_BACKEND

# @celery.task(name="train_ignite")
def train_task_model(
    conf_id: str,
    # user_id: str,
    # model_type: Optional[str] = None,
    # batch_size: int = Config.BATCH_SIZE,
    # num_epochs: int = Config.NUM_EPOCHS,
    # learning_rate: float = Config.LR,
    # momentum: float = Config.MOMENTUM,
    # weight_decay: float = Config.WEIGHT_DECAY,
    # optimizer_type: str = 'sgd',
    # data_root: str = Config.DATA_ROOT,
    # ann_file: Optional[str] = Config.ANN_FILE,
    # device: torch.device = Config.DEVICE
) -> Tuple[torch.nn.Module, Engine, TrainingHistory]:
    """
    Функция для запуска обучения модели в зависимости от типа задачи.
    
    Args:
        task_type (str): Тип задачи ('segmentation', 'detection', 'classification')
        model_type (str, optional): Тип модели. Если None, используется модель по умолчанию для задачи
        num_classes (int, optional): Количество классов. Если None, определяется автоматически
        batch_size (int): Размер батча
        num_epochs (int): Количество эпох
        learning_rate (float): Скорость обучения
        momentum (float): Моментум для SGD/RMSprop
        weight_decay (float): Вес регуляризации
        optimizer_type (str): Тип оптимизатора ('sgd', 'adam', 'adamw', 'rmsprop')
        data_root (str): Путь к директории с данными
        ann_file (str, optional): Путь к файлу аннотаций (требуется для segmentation и detection)
        device (torch.device): Устройство для обучения (CPU/GPU)
    
    Returns:
        tuple: (model, trainer, history)
            - model: обученная модель
            - trainer: объект trainer
            - history: объект TrainingHistory с историей метрик
    """


    # Устанавливаем модель по умолчанию, если не указана
    # if model_type is None:
    #     if task_type == 'segmentation':
    #         model_type = 'unet'
    #     elif task_type == 'detection':
    #         model_type = 'fasterrcnn'
    #     elif task_type == 'classification':
    #         model_type = 'resnet50'
    
    db = _session()
    
    train_info = db.query(TrainingConfiguration).filter_by(id=conf_id).first()
    # conf = train_info.training_conf
    train_info.status = 'processing'
    db.commit()

    logging.info("starting train")
    print('-----')
    # print(train_info.training_conf)
    print(train_info.training_conf['task_type'])
    print('-----')

    try:
      with TemporaryDirectory(dir=os.getcwd()) as tmp:
        # Подготавливаем DataLoader
        logging.info("starting download dataset")
        with open(tmp + train_info.dataset_s3_location, mode='w+b') as f:
            print(train_info.dataset_s3_location),
            bucket='cvat'
            # bucket=f"user-{user_id}"
            print(bucket)

            s3.download_file(f, train_info.dataset_s3_location, bucket)

        # print(f'task_type: {conf["taks_type"]}')

        logging.info("starting extract data")
        print('----------')
        print(f'type: {train_info.training_conf["task_type"]}')
        print('----------')
        data_root, ann_file = unzip_dataset(
          tmp, 
          location=train_info.dataset_s3_location, 
          task_type=train_info.training_conf['task_type']
        )

        print('----------')
        print(f'root: {data_root}')
        print(f'ann: {ann_file}')
        print('----------')

        train_loader, classes = prepare_dataloader(
          task_type=train_info.training_conf['task_type'], 
          data_root=data_root, 
          ann_file=ann_file, 
          batch_size=train_info.training_conf['batch'],
        )
        print(train_loader)


        # with TemporaryDirectory(dir=os.getcwd()) as tmp:
        print('here', train_info.training_conf)
        if train_info.training_conf['task_type'] == 'segmentation':
              model, trainer, history = train_segmentation_model(
                  model_type=train_info.model,
                  num_classes=get_num_classes_from_annotations(ann_file),
                  num_epochs=train_info.training_conf['epochs'],
                  learning_rate=BaseConfig.LR,
                  momentum=BaseConfig.MOMENTUM,
                  weight_decay=BaseConfig.WEIGHT_DECAY,
                  optimizer_type=train_info.training_conf['optimizer'],
                  train_loader=train_loader,
                  # device=BaseConfig.DEVICE
              )
        elif train_info.training_conf['task_type'] == 'detection':
            model, trainer, history = train_detection_model(
                model_type=train_info.model,
                num_classes=get_num_classes_from_annotations(ann_file),
                num_epochs=train_info.training_conf['epochs'],
                learning_rate=BaseConfig.LR,
                momentum=BaseConfig.MOMENTUM,
                weight_decay=BaseConfig.MOMENTUM,
                optimizer_type=train_info.training_conf['optimizer'],
                train_loader=train_loader,
                # device=BaseConfig.DEVICE
            )
        elif train_info.training_conf['task_type'] == 'classification':
            model, trainer, history = train_classification_model(
                model_type=train_info.model,
                num_classes=len(classes),
                num_epochs=train_info.training_conf['epochs'],
                learning_rate=BaseConfig.LR,
                momentum=BaseConfig.MOMENTUM,
                weight_decay=BaseConfig.WEIGHT_DECAY,
                optimizer_type=train_info.training_conf['optimizer'],
                train_loader=train_loader,
                # device=BaseConfig.DEVICE
            )
        else:
            raise ValueError(f"Unknown task type: {train_info.training_conf['task_type']}")
        
        train_info.result_metrics = convert_numpy_types(history.get_history())
        db.commit()

        real_batch = next(iter(train_loader))[0]  
        dummy_input = real_batch[:1]      
        s3_location = save_model(
          tmp,
          conf_id,
          model_name=f'{train_info.model}-{conf_id}',
          model=model,
          bucket='cvat',
          dummy_input=dummy_input
        )

        train_info.weight_s3_location = s3_location
        train_info.status = 'processed'
        db.commit()

        print('------------')
        # print(model)
        # print(history.get_history())
        print('------------')

        return {
          'id': conf_id,
          'precision': train_info.result_metrics['best_metrics']['precision'],
          'recall': train_info.result_metrics['best_metrics']['recall'],
          'loss': train_info.result_metrics['best_metrics']['loss'],
        }
    except Exception as e:
      print(e)
      train_info.status = 'error'
      db.commit()  
    # Запускаем обучение в зависимости от типа задачи
  
# if __name__ == "__main__":
#     # Пример использования для сегментации
#     model, trainer, history = train_task_model(
#         task_type='segmentation',
#         model_type='unet',
#         optimizer_type='adam',
#         learning_rate=0.001
#     )
    
    # # Пример использования для детекции
    # model, trainer, history = train_task_model(
    #     task_type='detection',
    #     model_type='fasterrcnn',
    #     optimizer_type='sgd',
    #     learning_rate=0.005
    # )
    
    # # Пример использования для классификации
    # model, trainer, history = train_task_model(
    #     task_type='classification',
    #     model_type='resnet50',
    #     optimizer_type='sgd',
    #     learning_rate=0.005
    # ) 