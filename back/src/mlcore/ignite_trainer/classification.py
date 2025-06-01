import torch
import torchvision
from torchvision.models import resnet50, efficientnet_b0, mobilenet_v3_large
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy, Precision, Recall, ConfusionMatrix
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
import os
import glob
from typing import Literal, Tuple
import numpy as np

from torchvision.datasets import ImageFolder
from torchvision import transforms

from .utils import TrainingHistory, get_optimizer


# Конфигурация
class Config:
    MODEL_TYPE: Literal['resnet50', 'efficientnet', 'mobilenet'] = 'resnet50'  # Выбор модели
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
    NUM_WORKERS = 4  # Количество воркеров для загрузки данных

# Инициализация модели
def get_model(model_type: str, num_classes: int):
    if model_type == 'resnet50':
        model = resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'efficientnet':
        model = efficientnet_b0(pretrained=True)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'mobilenet':
        model = mobilenet_v3_large(pretrained=True)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(Config.DEVICE)

# Функция для обработки батча
def process_batch(batch):
    images, labels = batch
    images = images.to(Config.DEVICE)
    labels = labels.to(Config.DEVICE)
    return images, labels

def calculate_metrics(pred_labels, target_labels):
    # Преобразуем в numpy для вычислений
    pred_labels = pred_labels.cpu().numpy()
    target_labels = target_labels.cpu().numpy()
    
    # Вычисляем метрики
    correct = (pred_labels == target_labels).sum()
    total = len(target_labels)
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Вычисляем precision и recall для каждого класса
    precision = []
    recall = []
    
    for cls in range(Config.NUM_CLASSES):
        true_positives = ((pred_labels == cls) & (target_labels == cls)).sum()
        false_positives = ((pred_labels == cls) & (target_labels != cls)).sum()
        false_negatives = ((pred_labels != cls) & (target_labels == cls)).sum()
        
        class_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        class_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        precision.append(class_precision)
        recall.append(class_recall)
    
    # Усредняем метрики по классам
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': mean_precision,
        'recall': mean_recall,
        'f1_score': f1_score
    }

# Функция для обучения
def train_step(engine, batch, model, optimizer):
    model.train()
    images, labels = process_batch(batch)
    
    optimizer.zero_grad()
    outputs = model(images)
    
    # Вычисляем loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    # Проверяем значения loss на NaN
    if torch.isnan(loss):
        print("Warning: NaN loss detected!")
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    loss.backward()
    
    # Добавляем gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Вычисляем метрики
    with torch.no_grad():
        pred_labels = torch.argmax(outputs, dim=1)
        metrics = calculate_metrics(pred_labels, labels)
        metrics['loss'] = loss.item()
    
    return metrics

def get_transforms(is_train: bool = True) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE + 32),
            transforms.CenterCrop(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def init_metrics(trainer: Engine) -> None:
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['accuracy']).attach(trainer, 'accuracy')
    RunningAverage(output_transform=lambda x: x['precision']).attach(trainer, 'precision')
    RunningAverage(output_transform=lambda x: x['recall']).attach(trainer, 'recall')
    RunningAverage(output_transform=lambda x: x['f1_score']).attach(trainer, 'f1_score')
    
    ProgressBar(persist=True).attach(trainer, ['loss', 'accuracy', 'precision', 'recall', 'f1_score'])

def train_model(
    model_type: str = Config.MODEL_TYPE,
    num_classes: int = None,  # Теперь это опциональный параметр
    # batch_size: int = Config.BATCH_SIZE,
    num_epochs: int = Config.NUM_EPOCHS,
    train_loader: DataLoader = None,
    learning_rate: float = 0.001,
    momentum: float = Config.MOMENTUM,
    weight_decay: float = Config.WEIGHT_DECAY,
    optimizer_type: str = 'adam',
    # data_root: str = Config.DATA_ROOT,
    # device: torch.device = Config.DEVICE
):
    """
    Функция для запуска обучения модели классификации.
    
    Args:
        model_type (str): Тип модели ('resnet50', 'efficientnet' или 'mobilenet')
        num_classes (int, optional): Количество классов. Если None, определяется автоматически из структуры директорий
        batch_size (int): Размер батча
        num_epochs (int): Количество эпох
        learning_rate (float): Скорость обучения
        momentum (float): Моментум для SGD/RMSprop
        weight_decay (float): Вес регуляризации
        optimizer_type (str): Тип оптимизатора ('sgd', 'adam', 'adamw', 'rmsprop')
        data_root (str): Путь к корневой директории с данными
        device (torch.device): Устройство для обучения (CPU/GPU)
    
    Returns:
        tuple: (model, trainer, history)
            - model: обученная модель
            - trainer: объект trainer
            - history: объект TrainingHistory с историей метрик
    """
    # Инициализация модели
    model = get_model(model_type, num_classes)
    
    # Создаем оптимизатор выбранного типа
    optimizer = get_optimizer(
        model,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum
    )
    
    # Добавляем learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True
    )

    # Инициализация истории обучения
    history = TrainingHistory()

    # Настройка Ignite
    trainer = Engine(lambda engine, batch: train_step(engine, batch, model, optimizer))
    
    # Инициализируем метрики
    init_metrics(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        metrics = engine.state.metrics
        current_lr = optimizer.param_groups[0]['lr']
        
        # Обновляем историю
        history.update(metrics, current_lr)
        
        print(
            f"Epoch {engine.state.epoch}, "
            f"Loss: {metrics['loss']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1 Score: {metrics['f1_score']:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        
        # Обновляем learning rate на основе loss
        scheduler.step(metrics['loss'])

    # Запуск обучения
    trainer.run(train_loader, max_epochs=num_epochs)
    
    # Выводим полную историю обучения
    history.print_history()
    
    return model, trainer, history

# if __name__ == "__main__":
#     # Запуск обучения с автоматическим определением количества классов
#     model, trainer, history = train_model(
#         optimizer_type='adam',
#         learning_rate=0.001
#     ) 