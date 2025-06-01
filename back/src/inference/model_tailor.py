import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from torchvision.models import resnet50, efficientnet_b0, mobilenet_v3_large
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from mlcore.ignite_trainer.segmentation import UNet
from PIL import Image
from torchvision import transforms as F
import torch

from db.session import _session
from models.user import TrainingConfiguration
from s3.s3 import s3
from tempfile import TemporaryDirectory
import os

class Tailor:
    def __init__(self, model_type: str, is_custom: bool):
        self.model = self.sew_model(model_type,is_custom)
        self.task_type = self.sew_type(model_type,is_custom)

    def sew_model(self, model_type: str, is_custom: bool):
        if is_custom:
            db = _session()
            pipeline = db.query(TrainingConfiguration).filter_by(id=model_type).first()
            model_s3_path = '/' + pipeline.weight_s3_location.replace('/', '-')

            with TemporaryDirectory(dir=os.getcwd()) as tmp:
                try:
                    with open(tmp + model_s3_path, mode='w+b') as f:
                        bucket='cvat'
                        s3.download_file(f, pipeline.weight_s3_location, bucket)
                        loaded_model = torch.load(tmp + model_s3_path)
                        return loaded_model
                except Exception as e:
                    print(e)
            # with TemporaryDirectory(dir=os.getcwd()) as tmp:
            #     # Формируем полный путь к файлу в временной директории
            #     model_file_path = os.path.join(tmp, model_s3_path)

            #     # Открываем файл для записи
            #     print(f'Downloading {model_s3_path} from bucket to {tmp+model_file_path}', flush=True)
            #     with open(tmp+model_file_path, 'w+b') as f:
            #         bucket = "cvat"
            #         print(f'Downloading {model_s3_path} from bucket {bucket} to {model_file_path}', flush=True)
                    
            #         # Используем ваш метод для загрузки файла
            #         s3.download_file(f, model_s3_path, bucket)

            #         # Проверяем существование файла
            #         if os.path.exists(model_file_path):
            #             print(f"File exists: {model_file_path}")
            #             # Загружаем модель из файла
            #             loaded_model = torch.load(model_file_path)
            #             return loaded_model
            #         else:
            #             raise FileNotFoundError(f"File not found: {model_file_path}")

        if 'yolo' in model_type.lower():
            model = YOLO(str(model_type) + '.pt')
            return model
        else:
            if model_type == 'deeplabv3':
                model = deeplabv3_resnet50(pretrained=True)
            elif model_type == 'fcn':
                model = fcn_resnet50(pretrained=True)

            elif model_type == 'fasterrcnn':
                model = fasterrcnn_resnet50_fpn(pretrained=True)
            elif model_type == 'ssd':
                model = ssd300_vgg16(pretrained=True)

            elif model_type == 'resnet50':
                model = resnet50(pretrained=True)
            elif model_type == 'efficientnet':
                model = efficientnet_b0(pretrained=True)
            elif model_type == 'mobilenet':
                model = mobilenet_v3_large(pretrained=True)

            else:
                raise ValueError(f"Unknown model type: {model_type}")
            model.eval()
            return model
        
    def sew_type(self, model_type: str, is_custom: bool):
        model_init = model_type
        if is_custom:
            db = _session()
            pipeline = db.query(TrainingConfiguration).filter_by(id=model_type).first()
            model_init = pipeline.model
        
        print(f'model_init ${model_init}', flush=True)
        
        if 'yolo' in model_init.lower():
            return 'yolo'
        else:
            if model_init in ['deeplabv3', 'fcn']:
                return 'segmentation'
            elif model_init in ['fasterrcnn', 'ssd']: 
                return 'detection'
            elif model_init in ['resnet50', 'efficientnet', 'mobilenet']: 
                return 'classification'
            else:
                return None
    
    def predict(self, frame: np.ndarray):
        # Преобразование изображения из формата OpenCV в PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Если модель YOLO, используем другой подход
        if isinstance(self.model, YOLO):
            results = self.model(image)  # Получаем предсказания
            return results  # Возвращаем результаты YOLO

        # Подготовка изображения с автоматической трансформацией для других моделей
        to_tensor = F.ToTensor()  # Создаем экземпляр ToTensor
        image_tensor = to_tensor(image).unsqueeze(0)  # Преобразуем в тензор и добавляем размерность батча

        # Нормализация
        normalize = F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)  # Применяем нормализацию
        
        # Получение предсказания для других моделей
        with torch.no_grad():
            predictions = self.model(image_tensor)

        return predictions