import os
import os
import zipfile
import logging

import pandas as pd
from fastapi import Depends
from ultralytics import YOLO
from tempfile import TemporaryDirectory
from celery import Celery
from pathlib import Path

import os.path as p

from db import Session
from models.user import TrainingConfiguration
from s3.s3 import s3
from settings import settings
from db.session import _session
from db import get_database
from sqlalchemy.orm import Session

def is_image(filename):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    return any(filename.lower().endswith(ext) for ext in image_extensions)


def train_yolo(
        conf_id: int, 
        # user_id: int
      ):
    db = _session()
    conf = db.query(TrainingConfiguration).filter_by(id=conf_id).first()
    yolo = YOLO(str(conf.model) + '.yaml')
    conf.status = 'processing'
    bucket=f"cvat"

    db.commit()
    try:
        with TemporaryDirectory(dir=os.getcwd()) as tmp:


            logging.info("starting download dataset")
            with open(tmp + conf.dataset_s3_location, mode='w+b') as f:
                print(conf.dataset_s3_location),
                # print(f"user-{user_id}")
                # bucket=f"user-{user_id}"
                bucket=f"cvat"
                print(f'bucket: {bucket}')
                s3.download_file(f, conf.dataset_s3_location, bucket)
            
            
            logging.info("starting extract data")
            with zipfile.ZipFile(tmp + conf.dataset_s3_location, 'r') as f:
                print(f.namelist())
                f.extract('obj.names', path=tmp + '/data')
                [f.extract(file, path=tmp + '/data/images/') for file in f.namelist() if (is_image(file)) and file.startswith('obj_train_data')]
                [f.extract(file, path=tmp + '/data/images/') for file in f.namelist() if (is_image(file) and file.startswith('obj_validation_data'))]
                [f.extract(file, path=tmp + '/data/labels/') for file in f.namelist() if (file.endswith('.txt') and file.startswith('obj_train_data'))]
                [f.extract(file, path=tmp + '/data/labels/') for file in f.namelist() if (file.endswith('.txt') and file.startswith('obj_validation_data'))]

            path = Path(tmp + '/data/images/obj_train_data')
            path.rename(tmp + '/data/images/train')

            path = Path(tmp + '/data/labels/obj_train_data')
            path.rename(tmp + '/data/labels/train')

            print(os.listdir(tmp + '/data/images/train/'))
            logging.info("generating yaml file")
            classes = []
            with open(tmp + '/data/obj.names', 'r') as f:
                for line in f.readlines():
                    classes.append(line.replace('\n', ''))
            print(classes)
            
            
            training_conf = conf.training_conf
            training_conf['classes'] = classes
            conf.training_conf = training_conf
            db.commit()
            
            
            with open(tmp + '/data/data.yaml', 'w') as f:
                names = [f'  {classes.index(name)}: {name}\n' for name in classes]
                f.writelines(['path: ' + tmp + '/data/\n',
                            'train: images/train\n',
                            'val: images/train\n',
                            'names:\n'] + names)
            name = f'yolo-{conf.id}'
            result = yolo.train(
                data=tmp + '/data/data.yaml',
                project=tmp,
                name=name,
                epochs=conf.training_conf['epochs'],
                # patience=conf.training_conf['patience'],
                batch=conf.training_conf['batch'],
                # imgsz=conf.training_conf['imgsz'],
                optimizer=conf.training_conf['optimizer'],
                device=conf.training_conf['device'] if conf.training_conf['device'] == 'cpu' else 0
            )

            yolo.export(format='onnx')

            data = pd.read_csv(tmp + f'/{name}/results.csv')
            data = data.to_dict()
            clean_data = {}

            for key in data.keys():
                if key.strip() == 'metrics/precision(B)':
                    clean_data['precision'] = data[key]
                elif key.strip() == 'metrics/recall(B)':
                    clean_data['recall'] = data[key]
                elif key.strip() == 'train/box_loss':
                    clean_data['train/box_loss'] = data[key]
                elif key.strip() == 'train/cls_loss':
                    clean_data['train/cls_loss'] = data[key]
                elif key.strip() == 'train/dfl_loss':
                    clean_data['train/dfl_loss'] = data[key]
                elif key.strip() == 'metrics/mAP50(B)':
                    clean_data['mAP50'] = data[key]
                elif key.strip() == 'metrics/mAP50-95(B)':
                    clean_data['mAP50-95'] = data[key]
                elif key.strip() == 'val/box_loss':
                    clean_data['val/box_loss'] = data[key]
                elif key.strip() == 'val/cls_loss':
                    clean_data['val/cls_loss'] = data[key]
                elif key.strip() == 'val/dfl_loss':
                    clean_data['val/dfl_loss'] = data[key]

            conf.result_metrics = clean_data
            db.commit()

            with open(tmp + f'/{name}/weights/best.pt', 'rb') as f:
                path = f'/user/{conf_id}/result/best.pt'
                s3.upload_file(f, path, bucket)

            with open(tmp + f'/{name}/weights/best.onnx', 'rb') as f:
                path = f'/user/{conf_id}/result/best.onnx'
                s3.upload_file(f, path, bucket)
    
        conf.weight_s3_location = f'/user/{conf_id}/result/best.pt'
        conf.onnx_s3_location = f'/user/{conf_id}/result/best.onnx'

        conf.status = 'processed'
        db.commit()
        print('hehe')

        best_epoch = result.best_epoch
        best_metrics = {
            'loss': result.results[best_epoch]['metrics/mAP50(B)'],
            'precision': result.results[best_epoch]['metrics/precision(B)'],
            'recall': result.results[best_epoch]['metrics/recall(B)']
        }
        print(best_metrics)
        return {
          'id': conf_id,
          'precision': clean_data['precision'],
          'recall': clean_data['recall'],
          'loss': clean_data['train/box_loss'],
        }
    except Exception as e:
        print(e)
        conf.status = 'error'
        db.commit()
