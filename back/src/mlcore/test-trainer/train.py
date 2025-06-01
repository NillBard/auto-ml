import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any
from ignite.engine import Engine
from .seg import train_segmentation_model
from .detection import train_model as train_detection_model
from .classification import train_model as train_classification_model
from .utils import Config, get_optimizer, TrainingHistory

def prepare_dataloader(
    task_type: str,
    data_root: str,
    ann_file: Optional[str] = None,
    batch_size: int = Config.BATCH_SIZE
) -> DataLoader:
    """
    Подготавливает DataLoader в зависимости от типа задачи.
    
    Args:
        task_type (str): Тип задачи ('segmentation', 'detection', 'classification')
        data_root (str): Путь к директории с данными
        ann_file (str, optional): Путь к файлу аннотаций (требуется для segmentation и detection)
        batch_size (int): Размер батча
    
    Returns:
        DataLoader: Загрузчик данных для обучения
    """
    if task_type == 'segmentation':
        from .seg import prepare_dataloader as prepare_seg_dataloader
        return prepare_seg_dataloader(data_root, ann_file, batch_size)
    elif task_type == 'detection':
        from .detection import prepare_dataloader as prepare_det_dataloader
        return prepare_det_dataloader(data_root, ann_file, batch_size)
    elif task_type == 'classification':
        from .classification import prepare_dataloader as prepare_cls_dataloader
        return prepare_cls_dataloader(data_root, batch_size)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def train_task_model(
    task_type: str,
    model_type: Optional[str] = None,
    num_classes: Optional[int] = None,
    batch_size: int = Config.BATCH_SIZE,
    num_epochs: int = Config.NUM_EPOCHS,
    learning_rate: float = Config.LR,
    momentum: float = Config.MOMENTUM,
    weight_decay: float = Config.WEIGHT_DECAY,
    optimizer_type: str = 'sgd',
    data_root: str = Config.DATA_ROOT,
    ann_file: Optional[str] = Config.ANN_FILE,
    device: torch.device = Config.DEVICE
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
    if model_type is None:
        if task_type == 'segmentation':
            model_type = 'unet'
        elif task_type == 'detection':
            model_type = 'fasterrcnn'
        elif task_type == 'classification':
            model_type = 'resnet50'
    
    # Подготавливаем DataLoader
    train_loader = prepare_dataloader(task_type, data_root, ann_file, batch_size)
    
    # Запускаем обучение в зависимости от типа задачи
    if task_type == 'segmentation':
        return train_segmentation_model(
            model_type=model_type,
            num_classes=num_classes,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            train_loader=train_loader,
            device=device
        )
    elif task_type == 'detection':
        return train_detection_model(
            model_type=model_type,
            num_classes=num_classes,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            train_loader=train_loader,
            device=device
        )
    elif task_type == 'classification':
        return train_classification_model(
            model_type=model_type,
            num_classes=num_classes,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            train_loader=train_loader,
            device=device
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")

if __name__ == "__main__":
    # Пример использования для сегментации
    model, trainer, history = train_task_model(
        task_type='segmentation',
        model_type='unet',
        optimizer_type='adam',
        learning_rate=0.001
    )
    
    # Пример использования для детекции
    model, trainer, history = train_task_model(
        task_type='detection',
        model_type='fasterrcnn',
        optimizer_type='sgd',
        learning_rate=0.005
    )
    
    # Пример использования для классификации
    model, trainer, history = train_task_model(
        task_type='classification',
        model_type='resnet50',
        optimizer_type='sgd',
        learning_rate=0.005
    ) 