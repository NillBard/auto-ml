import torch
import torchvision
from torchvision.models import resnet50, efficientnet_b0, mobilenet_v3_large, resnet101, resnet152
from torch.utils.data import DataLoader
from torch.optim import SGD
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy, Precision, Recall, ConfusionMatrix
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
import os
import glob
from typing import Literal, Tuple
import numpy as np
from dataset_utils import prepare_classification_dataset
from torchvision.datasets import ImageFolder
from .utils import Config, get_optimizer, TrainingHistory

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
    elif model_type == 'resnet101':
        model = resnet101(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'resnet152':
        model = resnet152(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(Config.DEVICE)

# Функция для обработки батча
def process_batch(batch):
    """
    Обрабатывает батч данных.
    
    Args:
        batch: Кортеж (images, labels) из DataLoader
    
    Returns:
        tuple: (images, labels) на нужном устройстве
    """
    images, labels = batch
    images = images.to(Config.DEVICE)
    labels = labels.to(Config.DEVICE)
    return images, labels

def calculate_metrics(pred_labels, target_labels):
    """
    Вычисление метрик для классификации.
    
    Args:
        pred_labels: Предсказанные метки [N]
        target_labels: Целевые метки [N]
    
    Returns:
        dict: Словарь с метриками
    """
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

class TrainingHistory:
    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1_score = []
        self.learning_rates = []
    
    def update(self, metrics, lr):
        self.loss.append(metrics['loss'])
        self.accuracy.append(metrics['accuracy'])
        self.precision.append(metrics['precision'])
        self.recall.append(metrics['recall'])
        self.f1_score.append(metrics['f1_score'])
        self.learning_rates.append(lr)
    
    def get_best_epoch(self, metric='f1_score'):
        """Возвращает номер эпохи с лучшим значением указанной метрики"""
        if metric == 'loss':
            return np.argmin(self.loss)
        else:
            return np.argmax(getattr(self, metric))
    
    def get_best_metrics(self, metric='f1_score'):
        """Возвращает метрики лучшей эпохи"""
        best_epoch = self.get_best_epoch(metric)
        return {
            'epoch': best_epoch,
            'loss': self.loss[best_epoch],
            'accuracy': self.accuracy[best_epoch],
            'precision': self.precision[best_epoch],
            'recall': self.recall[best_epoch],
            'f1_score': self.f1_score[best_epoch],
            'learning_rate': self.learning_rates[best_epoch]
        }
    
    def get_history(self):
        """Возвращает полную историю обучения в виде словаря"""
        return {
            'epochs': list(range(len(self.loss))),
            'loss': self.loss,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'learning_rates': self.learning_rates,
            'best_metrics': self.get_best_metrics()
        }
    
    def print_history(self):
        """Выводит полную историю обучения в виде таблицы"""
        print("\nTraining History:")
        print("-" * 100)
        print(f"{'Epoch':^6} | {'Loss':^10} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10} | {'LR':^12}")
        print("-" * 100)
        
        for epoch in range(len(self.loss)):
            print(f"{epoch:^6} | {self.loss[epoch]:^10.4f} | {self.accuracy[epoch]:^10.4f} | "
                  f"{self.precision[epoch]:^10.4f} | {self.recall[epoch]:^10.4f} | "
                  f"{self.f1_score[epoch]:^10.4f} | {self.learning_rates[epoch]:^12.6f}")
        
        print("-" * 100)
        
        # Выводим лучшие значения
        best_metrics = self.get_best_metrics()
        print("\nBest Epoch Metrics (by F1 Score):")
        print(f"Epoch: {best_metrics['epoch']}")
        print(f"Loss: {best_metrics['loss']:.4f}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        print(f"F1 Score: {best_metrics['f1_score']:.4f}")
        print(f"Learning rate: {best_metrics['learning_rate']:.6f}")

def get_optimizer(model, optimizer_type: str, learning_rate: float, weight_decay: float, momentum: float = 0.9):
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

def get_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Создает трансформации для обучения или валидации.
    
    Args:
        is_train (bool): Если True, возвращает трансформации для обучения с аугментацией
    
    Returns:
        transforms.Compose: Набор трансформаций
    """
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

def prepare_dataloaders(data_root: str, batch_size: int, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Подготавливает DataLoader'ы для обучения и валидации.
    
    Args:
        data_root (str): Путь к корневой директории с данными
        batch_size (int): Размер батча
        val_split (float): Доля данных для валидации
    
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    # Создаем датасет
    full_dataset = ImageFolder(
        root=data_root,
        transform=get_transforms(is_train=True)  # Начальная трансформация для определения размеров
    )
    
    # Разделяем на train и val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Применяем соответствующие трансформации
    train_dataset.dataset.transform = get_transforms(is_train=True)
    val_dataset.dataset.transform = get_transforms(is_train=False)
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"Classes: {full_dataset.classes}")
    
    return train_loader, val_loader

def init_metrics(trainer: Engine) -> None:
    """
    Инициализирует метрики для обучения модели классификации.
    
    Args:
        trainer (Engine): Ignite engine для обучения
    """
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['accuracy']).attach(trainer, 'accuracy')
    RunningAverage(output_transform=lambda x: x['precision']).attach(trainer, 'precision')
    RunningAverage(output_transform=lambda x: x['recall']).attach(trainer, 'recall')
    RunningAverage(output_transform=lambda x: x['f1_score']).attach(trainer, 'f1_score')
    
    ProgressBar(persist=True).attach(trainer, ['loss', 'accuracy', 'precision', 'recall', 'f1_score'])

def train_model(
    model_type: str = 'resnet50',
    num_classes: int = None,
    num_epochs: int = Config.NUM_EPOCHS,
    learning_rate: float = Config.LR,
    momentum: float = Config.MOMENTUM,
    weight_decay: float = Config.WEIGHT_DECAY,
    optimizer_type: str = 'sgd',
    train_loader: DataLoader = None,
    device: torch.device = Config.DEVICE
):
    """
    Функция для запуска обучения модели классификации.
    
    Args:
        model_type (str): Тип модели ('resnet50', 'resnet101', 'resnet152')
        num_classes (int, optional): Количество классов
        num_epochs (int): Количество эпох
        learning_rate (float): Скорость обучения
        momentum (float): Моментум для SGD/RMSprop
        weight_decay (float): Вес регуляризации
        optimizer_type (str): Тип оптимизатора ('sgd', 'adam', 'adamw', 'rmsprop')
        train_loader (DataLoader): Загрузчик данных для обучения
        device (torch.device): Устройство для обучения (CPU/GPU)
    
    Returns:
        tuple: (model, trainer, history)
            - model: обученная модель
            - trainer: объект trainer
            - history: объект TrainingHistory с историей метрик
    """
    if train_loader is None:
        raise ValueError("train_loader is required")
    
    # Определяем количество классов, если не указано
    if num_classes is None:
        num_classes = len(train_loader.dataset.classes)
        print(f"Automatically determined number of classes: {num_classes}")
    
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
    
    # Настраиваем метрики
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
            f"F1: {metrics['f1']:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        
        # Обновляем learning rate на основе loss
        scheduler.step(metrics['loss'])

    # Запуск обучения
    trainer.run(train_loader, max_epochs=num_epochs)
    
    # Выводим полную историю обучения
    history.print_history()
    
    return model, trainer, history

def prepare_dataloader(data_root: str, batch_size: int) -> DataLoader:
    """
    Подготавливает DataLoader для обучения модели классификации.
    
    Args:
        data_root (str): Путь к директории с изображениями
        batch_size (int): Размер батча
    
    Returns:
        DataLoader: Загрузчик данных для классификации
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(
        root=data_root,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    return train_loader

if __name__ == "__main__":
    # Запуск обучения с автоматическим определением количества классов
    model, trainer, history = train_model(
        optimizer_type='adam',
        learning_rate=0.001
    ) 