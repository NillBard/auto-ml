import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
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
from .utils import TrainingHistory, get_optimizer, prepare_detection_dataset, get_num_classes_from_coco

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

def get_model(model_type: str, num_classes: int):
    if model_type == 'fasterrcnn':
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_type == 'ssd':
        model = ssd300_vgg16(pretrained=True)

        out_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        
        model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(Config.DEVICE)

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
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'mean_iou': 0.0,
            'map': 0.0
        }
    
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    total_gt_boxes = len(target_boxes)
    total_pred_boxes = len(pred_boxes)
    total_correct = 0
    total_iou = 0.0
    
    gt_matched = np.zeros(len(target_boxes), dtype=bool)
    
    for pred_idx, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        best_iou = 0.0
        best_gt_idx = -1
        
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
        
        if best_iou > 0.5 and best_gt_idx != -1:
            total_correct += 1
            total_iou += best_iou
            gt_matched[best_gt_idx] = True
    
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
def train_step(engine, batch, model, optimizer):
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
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
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
    
    avg_metrics = {name: np.mean(values) for name, values in metrics.items()}
    avg_metrics['loss'] = losses.item()
    
    return avg_metrics

def get_num_classes_from_annotations(ann_file: str) -> int:
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    print(annotations['categories'])
    categories = {cat['id'] for cat in annotations['categories']}
    
    return len(categories) + 1

def init_metrics(trainer: Engine) -> None:
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['precision']).attach(trainer, 'precision')
    RunningAverage(output_transform=lambda x: x['recall']).attach(trainer, 'recall')
    RunningAverage(output_transform=lambda x: x['mean_iou']).attach(trainer, 'mean_iou')
    RunningAverage(output_transform=lambda x: x['map']).attach(trainer, 'map')
    
    ProgressBar(persist=True).attach(trainer, ['loss', 'precision', 'recall', 'mean_iou', 'map'])

def train_model(
    model_type: str = Config.MODEL_TYPE,
    num_classes: int = None,  # Теперь это опциональный параметр
    num_epochs: int = Config.NUM_EPOCHS,
    train_loader: DataLoader = None,
    learning_rate: float = 0.001,
    momentum: float = Config.MOMENTUM,
    weight_decay: float = Config.WEIGHT_DECAY,
    optimizer_type: str = 'adam',
    # data_root: str = Config.DATA_ROOT,
    # ann_file: str = Config.ANN_FILE,
    # device: torch.device = Config.DEVICE
):
    """
    Функция для запуска обучения модели детекции объектов.
    
    Args:
        model_type (str): Тип модели ('fasterrcnn' или 'ssd')
        num_classes (int, optional): Количество классов (включая фон). Если None, определяется автоматически из аннотаций
        batch_size (int): Размер батча
        num_epochs (int): Количество эпох
        learning_rate (float): Скорость обучения
        momentum (float): Моментум для SGD/RMSprop
        weight_decay (float): Вес регуляризации
        optimizer_type (str): Тип оптимизатора ('sgd', 'adam', 'adamw', 'rmsprop')
        data_root (str): Путь к директории с изображениями
        ann_file (str): Путь к файлу аннотаций
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
    history = TrainingHistory('detection')

    trainer = Engine(lambda engine, batch: train_step(engine, batch, model, optimizer))
    
    init_metrics(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        metrics = engine.state.metrics
        current_lr = optimizer.param_groups[0]['lr']
        
        history.update(metrics, current_lr)
        
        print(
            f"Epoch {engine.state.epoch}, "
            f"Loss: {metrics['loss']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"Mean IoU: {metrics['mean_iou']:.4f}, "
            f"mAP: {metrics['map']:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        
        scheduler.step(metrics['loss'])

    trainer.run(train_loader, max_epochs=num_epochs)
    
    history.print_history()
    
    return model, trainer, history

# if __name__ == "__main__":
#     model, trainer, history = train_model(
#         optimizer_type='adam',
#         learning_rate=0.001
#     )