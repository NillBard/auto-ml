import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from torch.optim import SGD
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
import os
import glob
from typing import Literal
import numpy as np
import json
from dataset_utils import prepare_detection_dataset, get_num_classes_from_coco
from torchvision.datasets import CocoDetection
from .utils import Config, get_optimizer, TrainingHistory, get_num_classes_from_annotations

# Конфигурация
class Config:
    MODEL_TYPE = 'fasterrcnn'  # Выбор модели
    NUM_CLASSES = 2  # Ваше количество классов + фон
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    LR = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    DATA_ROOT = './my/images/default'
    ANN_FILE = './my/annotations/instances_default.json'

# Инициализация модели
def get_model(model_type: str, num_classes: int):
    if model_type == 'fasterrcnn':
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_type == 'ssd':
        # Инициализация SSD300 с VGG16
        model = ssd300_vgg16(pretrained=True)
        
        # Получаем количество выходных каналов для SSD
        # Новый способ получения out_channels для SSD
        out_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        
        # Создаем новый головной классификатор
        model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
    elif model_type == 'maskrcnn':
        model = maskrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(Config.DEVICE)

# Функция для обработки батча
def process_batch(batch):
    images, targets = batch
    images = list(image.to(Config.DEVICE) for image in images)
    
    formatted_targets = []
    for target in targets:
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in target:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
                areas.append(ann['area'])
                iscrowd.append(ann['iscrowd'])
        
        if len(boxes) == 0:
            formatted_targets.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=Config.DEVICE),
                'labels': torch.zeros(0, dtype=torch.int64, device=Config.DEVICE),
                'area': torch.zeros(0, dtype=torch.float32, device=Config.DEVICE),
                'iscrowd': torch.zeros(0, dtype=torch.int64, device=Config.DEVICE)
            })
        else:
            formatted_targets.append({
                'boxes': torch.as_tensor(boxes, dtype=torch.float32, device=Config.DEVICE),
                'labels': torch.as_tensor(labels, dtype=torch.int64, device=Config.DEVICE),
                'area': torch.as_tensor(areas, dtype=torch.float32, device=Config.DEVICE),
                'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64, device=Config.DEVICE)
            })
    
    return images, formatted_targets

def calculate_metrics(pred_boxes, pred_labels, pred_scores, target_boxes, target_labels):
    """
    Вычисление метрик для детекции объектов.
    
    Args:
        pred_boxes: Предсказанные боксы [N, 4]
        pred_labels: Предсказанные метки [N]
        pred_scores: Предсказанные уверенности [N]
        target_boxes: Целевые боксы [M, 4]
        target_labels: Целевые метки [M]
    
    Returns:
        dict: Словарь с метриками
    """
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'mean_iou': 0.0,
            'map': 0.0
        }
    
    # Сортируем предсказания по уверенности
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Инициализация метрик
    total_gt_boxes = len(target_boxes)
    total_pred_boxes = len(pred_boxes)
    total_correct = 0
    total_iou = 0.0
    
    # Отслеживаем сопоставленные GT боксы
    gt_matched = np.zeros(len(target_boxes), dtype=bool)
    
    # Для каждого предсказания
    for pred_idx, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        best_iou = 0.0
        best_gt_idx = -1
        
        # Ищем лучший IoU среди GT боксов того же класса
        for gt_idx, (gt_box, gt_label) in enumerate(zip(target_boxes, target_labels)):
            if gt_label == pred_label and not gt_matched[gt_idx]:
                # Вычисляем IoU
                x1 = max(pred_box[0], gt_box[0])
                y1 = max(pred_box[1], gt_box[1])
                x2 = min(pred_box[2], gt_box[2])
                y2 = min(pred_box[3], gt_box[3])
                
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                union = pred_area + gt_area - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        # Если нашли хорошее соответствие (IoU > 0.5)
        if best_iou > 0.5 and best_gt_idx != -1:
            total_correct += 1
            total_iou += best_iou
            gt_matched[best_gt_idx] = True
    
    # Вычисляем метрики
    precision = total_correct / total_pred_boxes if total_pred_boxes > 0 else 0.0
    recall = total_correct / total_gt_boxes if total_gt_boxes > 0 else 0.0
    mean_iou = total_iou / total_correct if total_correct > 0 else 0.0
    map = precision * recall
    
    return {
        'precision': precision,
        'recall': recall,
        'mean_iou': mean_iou,
        'map': map
    }

# Функция для обучения
def train_step(engine, batch, model, optimizer, model_type):
    model.train()
    images, targets = process_batch(batch)
    
    optimizer.zero_grad()
    loss_dict = model(images, targets)
    
    # Проверяем значения loss на NaN
    losses = sum(loss for loss in loss_dict.values())
    if torch.isnan(losses):
        print("Warning: NaN loss detected!")
        return {
            'loss': float('inf'),
            'precision': 0.0,
            'recall': 0.0,
            'mean_iou': 0.0,
            'map': 0.0
        }
    
    losses.backward()
    
    # Добавляем gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Вычисляем метрики
    metrics = {}
    with torch.no_grad():
        model.eval()
        predictions = model(images)
        
        for pred, target in zip(predictions, targets):
            pred_metrics = calculate_metrics(
                pred['boxes'].cpu().numpy(),
                pred['labels'].cpu().numpy(),
                pred['scores'].cpu().numpy(),
                target['boxes'].cpu().numpy(),
                target['labels'].cpu().numpy()
            )
            
            for metric_name, value in pred_metrics.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(value)
    
    # Усредняем метрики по батчу
    avg_metrics = {name: np.mean(values) for name, values in metrics.items()}
    avg_metrics['loss'] = losses.item()
    
    return avg_metrics

def init_metrics(trainer: Engine) -> None:
    """
    Инициализирует метрики для обучения модели детекции объектов.
    
    Args:
        trainer (Engine): Ignite engine для обучения
    """
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['precision']).attach(trainer, 'precision')
    RunningAverage(output_transform=lambda x: x['recall']).attach(trainer, 'recall')
    RunningAverage(output_transform=lambda x: x['mean_iou']).attach(trainer, 'mean_iou')
    RunningAverage(output_transform=lambda x: x['map']).attach(trainer, 'map')
    
    ProgressBar(persist=True).attach(trainer, ['loss', 'precision', 'recall', 'mean_iou', 'map'])

def train_model(
    model_type: str = 'fasterrcnn',
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
    Функция для запуска обучения модели детекции.
    
    Args:
        model_type (str): Тип модели ('fasterrcnn' или 'maskrcnn')
        num_classes (int, optional): Количество классов (включая фон)
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
        num_classes = get_num_classes_from_annotations(Config.ANN_FILE)
        print(f"Automatically determined number of classes: {num_classes} (including background)")
    
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
    trainer = Engine(lambda engine, batch: train_step(engine, batch, model, optimizer, model_type))
    
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
            f"Classification Loss: {metrics['loss_classifier']:.4f}, "
            f"Box Loss: {metrics['loss_box_reg']:.4f}, "
            f"Objectness Loss: {metrics['loss_objectness']:.4f}, "
            f"RPN Box Loss: {metrics['loss_rpn_box_reg']:.4f}, "
            f"mAP: {metrics['map']:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        
        # Обновляем learning rate на основе loss
        scheduler.step(metrics['loss'])

    # Запуск обучения
    trainer.run(train_loader, max_epochs=num_epochs)
    
    # Выводим полную историю обучения
    history.print_history()
    
    return model, trainer, history

def prepare_dataloader(data_root: str, ann_file: str, batch_size: int) -> DataLoader:
    """
    Подготавливает DataLoader для обучения модели детекции.
    
    Args:
        data_root (str): Путь к директории с изображениями
        ann_file (str): Путь к файлу аннотаций
        batch_size (int): Размер батча
    
    Returns:
        DataLoader: Загрузчик данных для детекции
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CocoDetection(
        root=data_root,
        annFile=ann_file,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    return train_loader

if __name__ == "__main__":
    # Запуск обучения с автоматическим определением количества классов
    model, trainer, history = train_model(
        optimizer_type='adam',
        learning_rate=0.001
    )