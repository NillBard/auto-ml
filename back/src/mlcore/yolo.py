import os
import zipfile

import pandas as pd
from ultralytics import YOLO
from tempfile import TemporaryDirectory

import os.path as p

from db import Session
from models.user import TrainingConfiguration
from s3.s3 import s3


async def train_yolo(conf_id: int,
                     db: Session):
    conf = db.query(TrainingConfiguration).filter_by(id=conf_id).first()
    if conf is None:
        raise
    yolo = YOLO(str(conf.model) + '.yaml')
    conf.status = 'processing'
    db.commit()
    with TemporaryDirectory(dir=os.getcwd()) as tmp:

        with open(tmp + f'/{conf.name}.zip', mode='w+b') as f:
            s3.download_file(f, conf.dataset_s3_location)

        with zipfile.ZipFile(tmp + f'/{conf.name}.zip', 'r') as f:
            f.extractall(path=tmp + '/dataset')

        with open(tmp + '/dataset/data.yaml', 'w') as f:
            classes = conf.training_conf['classes']
            names = [f'  {classes.index(name)}: {name}\n' for name in classes]
            f.writelines(['path: ' + tmp + '/dataset\n',
                          'train: train/images\n',
                          'val: val/images\n',
                          'names:\n'] + names)

        yolo.train(
            data=tmp + '/dataset/data.yaml',
            project=tmp,
            name=conf.name,
            epochs=conf.training_conf['epochs'],
            patience=conf.training_conf['patience'],
            batch=conf.training_conf['batch'],
            imgsz=conf.training_conf['imgsz'],
            optimizer=conf.training_conf['optimizer'],
            device=conf.training_conf['device'] if conf.training_conf['device'] == 'cpu' else 0
        )

        yolo.export(format='onnx')

        data = pd.read_csv(tmp + f'/{conf.name}/results.csv')
        data = data.to_dict()
        clean_data = {}

        for key in data.keys():
            if key.strip() == 'metrics/precision(B)':
                clean_data['precision'] = data[key]
            elif key.strip() == 'metrics/recall(B)':
                clean_data['recall'] = data[key]

        conf.result_metrics = clean_data
        db.commit()

        with open(tmp + f'/{conf.name}/weights/best.pt', 'rb') as f:
            path = f'/user/{conf_id}/result/best.pt'
            s3.upload_file(f, path)

        with open(tmp + f'/{conf.name}/weights/best.onnx', 'rb') as f:
            path = f'/user/{conf_id}/result/best.onnx'
            s3.upload_file(f, path)

    conf.weight_s3_location = f'/user/{conf_id}/result/best.pt'
    conf.onnx_s3_location = f'/user/{conf_id}/result/best.onnx'

    conf.status = 'processed'
    db.commit()