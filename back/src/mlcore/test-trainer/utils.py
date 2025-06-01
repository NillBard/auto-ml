import torch
import torch.nn as nn
import numpy as np
import json
from typing import Optional, Dict, Any

class Config:
    """Общие настройки для всех типов задач"""
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    LR = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    DATA_ROOT = './coco-seg/images/default'
    ANN_FILE = './coco-seg/annotations/instances_default.json'

def get_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9
) -> torch.optim.Optimizer:
    """
    Создает оптимизатор указанного типа.
    
    Args:
        model: Модель для оптимизации
        optimizer_type (str): Тип оптимизатора ('sgd', 'adam', 'adamw', 'rmsprop')
        learning_rate (float): Скорость обучения
        weight_decay (float): Вес регуляризации
        momentum (float): Моментум (используется для SGD и RMSprop)
    
    Returns:
        optimizer: Оптимизатор PyTorch
    """
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

class TrainingHistory:
    """Класс для отслеживания истории обучения"""
    def __init__(self):
        self.loss = []
        self.precision = []
        self.recall = []
        self.mean_iou = []
        self.map = []
        self.learning_rates = []
    
    def update(self, metrics: Dict[str, float], lr: float) -> None:
        """Обновляет историю метриками из текущей эпохи"""
        self.loss.append(metrics['loss'])
        self.precision.append(metrics['precision'])
        self.recall.append(metrics['recall'])
        self.mean_iou.append(metrics['mean_iou'])
        self.map.append(metrics['map'])
        self.learning_rates.append(lr)
    
    def get_best_epoch(self, metric: str = 'map') -> int:
        """Возвращает номер эпохи с лучшим значением указанной метрики"""
        if metric == 'loss':
            return np.argmin(self.loss)
        else:
            values = getattr(self, metric)
            if not values:  # Если список пустой
                return 0
            return np.argmax(values)
    
    def get_best_metrics(self, metric: str = 'map') -> Dict[str, Any]:
        """Возвращает метрики лучшей эпохи"""
        best_epoch = self.get_best_epoch(metric)
        if best_epoch >= len(self.loss):  # Проверка на выход за границы
            best_epoch = 0
        
        return {
            'epoch': best_epoch,
            'loss': self.loss[best_epoch],
            'precision': self.precision[best_epoch],
            'recall': self.recall[best_epoch],
            'mean_iou': self.mean_iou[best_epoch],
            'map': self.map[best_epoch],
            'learning_rate': self.learning_rates[best_epoch]
        }
    
    def get_history(self) -> Dict[str, Any]:
        """Возвращает полную историю обучения в виде словаря"""
        return {
            'epochs': list(range(len(self.loss))),
            'loss': self.loss,
            'precision': self.precision,
            'recall': self.recall,
            'mean_iou': self.mean_iou,
            'map': self.map,
            'learning_rates': self.learning_rates,
            'best_metrics': self.get_best_metrics()
        }
    
    def print_history(self) -> None:
        """Выводит полную историю обучения в виде таблицы"""
        print("\nTraining History:")
        print("-" * 100)
        print(f"{'Epoch':^6} | {'Loss':^10} | {'Precision':^10} | {'Recall':^10} | {'Mean IoU':^10} | {'mAP':^10} | {'LR':^12}")
        print("-" * 100)
        
        for epoch in range(len(self.loss)):
            print(f"{epoch:^6} | {self.loss[epoch]:^10.4f} | {self.precision[epoch]:^10.4f} | "
                  f"{self.recall[epoch]:^10.4f} | {self.mean_iou[epoch]:^10.4f} | "
                  f"{self.map[epoch]:^10.4f} | {self.learning_rates[epoch]:^12.6f}")
        
        print("-" * 100)
        
        # Выводим лучшие значения
        best_metrics = self.get_best_metrics()
        print("\nBest Epoch Metrics (by mAP):")
        print(f"Epoch: {best_metrics['epoch']}")
        print(f"Loss: {best_metrics['loss']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        print(f"Mean IoU: {best_metrics['mean_iou']:.4f}")
        print(f"mAP: {best_metrics['map']:.4f}")
        print(f"Learning rate: {best_metrics['learning_rate']:.6f}")

def get_num_classes_from_annotations(ann_file: str) -> int:
    """
    Определяет количество классов из файла аннотаций COCO.
    
    Args:
        ann_file (str): Путь к файлу аннотаций в формате COCO
    
    Returns:
        int: Количество классов (включая фон)
    """
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Получаем уникальные категории
    categories = {cat['id'] for cat in annotations['categories']}
    
    # Возвращаем количество классов + 1 (для фона)
    return len(categories) + 1 