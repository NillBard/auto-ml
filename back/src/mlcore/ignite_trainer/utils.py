import os
from tempfile import TemporaryDirectory
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from typing import Tuple, Dict, Any
import json
import numpy as np
from s3.s3 import s3

class Config:
    NUM_CLASSES = 2  # Ваше количество классов
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    DATA_ROOT = './my/images/default'  # Путь к корневой директории с данными
    VAL_SPLIT = 0.2  # Доля данных для валидации
    IMAGE_SIZE = 224  # Размер изображения
    NUM_WORKERS = 0  # Количество воркеров для загрузки данных


class TrainingHistory:
    """Класс для отслеживания истории обучения"""
    def __init__(self, task_type: str = 'classification'):
        """
        Инициализация истории обучения.
        
        Args:
            task_type (str): Тип задачи ('classification', 'detection', 'segmentation')
        """
        self.task_type = task_type
        self.loss = []
        self.learning_rates = []
        
        # Общие метрики для всех задач
        self.precision = []
        self.recall = []
        
        # Специфичные метрики для каждого типа задачи
        if task_type == 'classification':
            self.accuracy = []
            self.f1_score = []
        elif task_type in ['detection', 'segmentation']:
            self.mean_iou = []
            self.map = []
            if task_type == 'segmentation':
                self.iou = []
    
    def update(self, metrics: Dict[str, float], lr: float) -> None:
        """Обновляет историю метриками из текущей эпохи"""
        self.loss.append(metrics['loss'])
        self.learning_rates.append(lr)
        self.precision.append(metrics['precision'])
        self.recall.append(metrics['recall'])
        
        if self.task_type == 'classification':
            self.accuracy.append(metrics['accuracy'])
            self.f1_score.append(metrics['f1_score'])
        elif self.task_type in ['detection', 'segmentation']:
            self.mean_iou.append(metrics['mean_iou'])
            self.map.append(metrics['map'])
            if self.task_type == 'segmentation':
                self.iou.append(metrics['iou'])
    
    def get_best_epoch(self, metric: str = None) -> int:
        """Возвращает номер эпохи с лучшим значением указанной метрики"""
        if metric is None:
            # Выбираем метрику по умолчанию в зависимости от типа задачи
            metric = 'f1_score' if self.task_type == 'classification' else 'map'
        
        if metric == 'loss':
            return np.argmin(self.loss)
        else:
            values = getattr(self, metric)
            if not values:  # Если список пустой
                return 0
            return np.argmax(values)
    
    def get_best_metrics(self, metric: str = None) -> Dict[str, Any]:
        """Возвращает метрики лучшей эпохи"""
        best_epoch = self.get_best_epoch(metric)
        if best_epoch >= len(self.loss):  # Проверка на выход за границы
            best_epoch = 0
        
        metrics = {
            'epoch': best_epoch,
            'loss': self.loss[best_epoch],
            'precision': self.precision[best_epoch],
            'recall': self.recall[best_epoch],
            'learning_rate': self.learning_rates[best_epoch]
        }
        
        if self.task_type == 'classification':
            metrics.update({
                'accuracy': self.accuracy[best_epoch],
                'f1_score': self.f1_score[best_epoch]
            })
        elif self.task_type in ['detection', 'segmentation']:
            metrics.update({
                'mean_iou': self.mean_iou[best_epoch],
                'map': self.map[best_epoch]
            })
            if self.task_type == 'segmentation':
                metrics['iou'] = self.iou[best_epoch]
        
        return metrics
    
    def get_history(self) -> Dict[str, Any]:
        """Возвращает полную историю обучения в виде словаря"""
        history = {
            'epochs': list(range(len(self.loss))),
            'loss': self.loss,
            'precision': self.precision,
            'recall': self.recall,
            'learning_rates': self.learning_rates,
            'best_metrics': self.get_best_metrics()
        }
        
        if self.task_type == 'classification':
            history.update({
                'accuracy': self.accuracy,
                'f1_score': self.f1_score
            })
        elif self.task_type in ['detection', 'segmentation']:
            history.update({
                'mean_iou': self.mean_iou,
                'map': self.map
            })
            if self.task_type == 'segmentation':
                history['iou'] = self.iou
        
        return history
    
    def print_history(self) -> None:
        """Выводит полную историю обучения в виде таблицы"""
        print("\nTraining History:")
        print("-" * 100)
        
        # Формируем заголовок таблицы в зависимости от типа задачи
        header = f"{'Epoch':^6} | {'Loss':^10} | {'Precision':^10} | {'Recall':^10}"
        if self.task_type == 'classification':
            header += f" | {'Accuracy':^10} | {'F1 Score':^10}"
        elif self.task_type in ['detection', 'segmentation']:
            header += f" | {'Mean IoU':^10} | {'mAP':^10}"
            if self.task_type == 'segmentation':
                header += f" | {'IoU':^10}"
        header += f" | {'LR':^12}"
        
        print(header)
        print("-" * 100)
        
        # Выводим метрики для каждой эпохи
        for epoch in range(len(self.loss)):
            metrics = f"{epoch:^6} | {self.loss[epoch]:^10.4f} | {self.precision[epoch]:^10.4f} | {self.recall[epoch]:^10.4f}"
            
            if self.task_type == 'classification':
                metrics += f" | {self.accuracy[epoch]:^10.4f} | {self.f1_score[epoch]:^10.4f}"
            elif self.task_type in ['detection', 'segmentation']:
                metrics += f" | {self.mean_iou[epoch]:^10.4f} | {self.map[epoch]:^10.4f}"
                if self.task_type == 'segmentation':
                    metrics += f" | {self.iou[epoch]:^10.4f}"
            
            metrics += f" | {self.learning_rates[epoch]:^12.6f}"
            print(metrics)
        
        print("-" * 100)
        
        # Выводим лучшие значения
        best_metrics = self.get_best_metrics()
        metric_name = 'F1 Score' if self.task_type == 'classification' else 'mAP'
        print(f"\nBest Epoch Metrics (by {metric_name}):")
        print(f"Epoch: {best_metrics['epoch']}")
        print(f"Loss: {best_metrics['loss']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        
        if self.task_type == 'classification':
            print(f"Accuracy: {best_metrics['accuracy']:.4f}")
            print(f"F1 Score: {best_metrics['f1_score']:.4f}")
        elif self.task_type in ['detection', 'segmentation']:
            print(f"Mean IoU: {best_metrics['mean_iou']:.4f}")
            print(f"mAP: {best_metrics['map']:.4f}")
            if self.task_type == 'segmentation':
                print(f"IoU: {best_metrics['iou']:.4f}")
        
        print(f"Learning rate: {best_metrics['learning_rate']:.6f}")

def get_optimizer(model, optimizer_type: str, learning_rate: float, weight_decay: float, momentum: float = 0.9):
    if optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. "
                        f"Supported types are: 'sgd', 'adam', 'adamw', 'rmsprop'")


def prepare_classification_dataset(
    data_root: str,
    batch_size: int,
    val_split: float = 0.2,
    image_size: int = 224,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    # Создаем трансформации
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Создаем датасет
    full_dataset = ImageFolder(
        root=data_root,
        transform=train_transform
    )
    
    # Разделяем на train и val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Применяем соответствующие трансформации
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=num_workers,
        pin_memory=True
    )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     # num_workers=num_workers,
    #     pin_memory=True
    # )
    
    classes = full_dataset.classes
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"Classes: {full_dataset.classes}")
    
    return train_loader, classes

def prepare_detection_dataset(
    data_root: str,
    ann_file: str,
    batch_size: int,
    num_workers: int = 4
) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = torchvision.datasets.CocoDetection(
        root=data_root,
        annFile=ann_file,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    classes = get_classes_from_coco(ann_file=ann_file)

    return loader, classes

def prepare_segmentation_dataloader(data_root: str, ann_file: str, batch_size: int):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.CocoDetection(
        root=data_root,
        annFile=ann_file,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    classes = get_classes_from_coco(ann_file=ann_file)
    
    return train_loader, classes

def get_classes_from_coco(ann_file: str) -> int:
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Получаем уникальные категории
    categories = [cat['name'] for cat in annotations['categories']]
    
    # Возвращаем количество классов + 1 (для фона в детекции/сегментации)
    return categories

def get_num_classes_from_coco(ann_file: str) -> int:
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Получаем уникальные категории
    categories = {cat['id'] for cat in annotations['categories']}
    
    # Возвращаем количество классов + 1 (для фона в детекции/сегментации)
    return len(categories) + 1 

def save_model(tmp, conf_id:str, model_name:str, model, bucket: str, dummy_input):
    torch_path=tmp + f'models/{model_name}.pt'
    onnx_path=tmp + f'models/{model_name}.onnx'
    # save_path = 'models/resnet18/weights.pt'

    # Создаём папки (если их нет)
    os.makedirs(os.path.dirname(torch_path), exist_ok=True)

    torch.save(model, torch_path)

    # torch.onnx.export(
    #   model,
    #   dummy_input,
    #   onnx_path,
    #   input_names=["input"],
    #   output_names=["output"],
    #   # dynamic_axes=dynamic_axes or {},
    #   verbose=False
    # )
  
    with open(torch_path, 'rb') as f:
      path = f'/user/{conf_id}/result/best.pt'
      s3.upload_file(f, path, bucket)
    return f'/user/{conf_id}/result/best.pt'
    # with open(onnx_path, 'rb') as f:
    #   path = f'/user/{conf_id}/result/best.onnx'
    #   s3.upload_file(f, path, bucket)

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj
  # def prepare_segmentation_dataset(
  #     data_root: str,
  #     ann_file: str,
  #     batch_size: int,
  #     num_workers: int = 4
  # ) -> DataLoader:
  #     transform = transforms.Compose([
  #         transforms.ToTensor(),
  #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  #     ])
      
  #     dataset = torchvision.datasets.CocoDetection(
  #         root=data_root,
  #         annFile=ann_file,
  #         transform=transform
  #     )
      
  #     loader = DataLoader(
  #         dataset,
  #         batch_size=batch_size,
  #         shuffle=True,
  #         num_workers=num_workers,
  #         collate_fn=lambda x: tuple(zip(*x))
  #     )
      
  #     print(f"Dataset size: {len(dataset)}")
  #     print(f"Number of classes: {len(dataset.coco.cats)}")
  #     print(f"Classes: {[cat['name'] for cat in dataset.coco.cats.values()]}")
      
  #     return loader

