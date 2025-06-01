import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
import json
import numpy as np
from .utils import Config, get_optimizer, TrainingHistory, get_num_classes_from_annotations

class Config:
    NUM_CLASSES = 2  # Ваше количество классов + фон
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    LR = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    DATA_ROOT = './coco-seg/images/default'
    ANN_FILE = './coco-seg/annotations/instances_default.json'


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

class TrainingHistory:
    def __init__(self):
        self.loss = []
        self.precision = []
        self.recall = []
        self.mean_iou = []
        self.map = []
        self.learning_rates = []
    
    def update(self, metrics, lr):
        self.loss.append(metrics['loss'])
        self.precision.append(metrics['precision'])
        self.recall.append(metrics['recall'])
        self.mean_iou.append(metrics['mean_iou'])
        self.map.append(metrics['map'])
        self.learning_rates.append(lr)
    
    def get_best_epoch(self, metric='map'):
        """Возвращает номер эпохи с лучшим значением указанной метрики"""
        if metric == 'loss':
            # Для loss ищем минимальное значение
            return np.argmin(self.loss)
        else:
            # Для остальных метрик ищем максимальное значение
            values = getattr(self, metric)
            if not values:  # Если список пустой
                return 0
            return np.argmax(values)
    
    def get_best_metrics(self, metric='map'):
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
    
    def get_history(self):
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
    
    def print_history(self):
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

class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._block(512, 1024)
        
        # Decoder
        self.upconv4 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(1024, 512)
        self.upconv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        self.upconv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        
        # Final convolution
        self.final = torch.nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(torch.nn.functional.max_pool2d(enc1, 2))
        enc3 = self.enc3(torch.nn.functional.max_pool2d(enc2, 2))
        enc4 = self.enc4(torch.nn.functional.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(torch.nn.functional.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

def get_segmentation_model(model_type: str, num_classes: int):
    """
    Создает модель сегментации указанного типа.
    
    Args:
        model_type (str): Тип модели ('deeplabv3', 'fcn', 'unet' или 'maskrcnn')
        num_classes (int): Количество классов (включая фон)
    
    Returns:
        model: Модель сегментации
    """
    if model_type == 'deeplabv3':
        model = deeplabv3_resnet50(pretrained=True)
        model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    elif model_type == 'fcn':
        model = fcn_resnet50(pretrained=True)
        model.classifier[-1] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
    elif model_type == 'unet':
        model = UNet(in_channels=3, out_channels=num_classes)
    elif model_type == 'maskrcnn':
        model = maskrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    else:
        raise ValueError(f"Unknown segmentation model type: {model_type}")
    
    return model.to(Config.DEVICE)

def is_point_in_polygon(point, polygon):
    """
    Проверяет, находится ли точка внутри полигона.
    Использует алгоритм ray casting.
    """
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def create_mask_from_polygon(polygon, size):
    """
    Создает бинарную маску из полигона.
    """
    mask = torch.zeros(size, dtype=torch.float32)
    h, w = size
    
    # Преобразуем полигон в список точек
    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
    
    print(f"\nСоздание маски из полигона:")
    print(f"Размер маски: {size}")
    print(f"Количество точек в полигоне: {len(points)}")
    print(f"Координаты полигона: {points[:5]}...")  # Показываем первые 5 точек
    
    # Используем более эффективный способ создания маски
    # Создаем сетку координат
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    points = torch.tensor(points, dtype=torch.float32)
    
    # Проверяем каждую точку сетки
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        
        # Проверяем, находится ли точка внутри полигона
        # Используем векторное произведение для определения ориентации
        cross_product = (p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0])
        mask[cross_product >= 0] = 1.0
    
    print(f"Значения в созданной маске: min={mask.min().item()}, max={mask.max().item()}")
    print(f"Количество ненулевых элементов: {(mask > 0).sum().item()}")
    
    return mask

def process_segmentation_batch(batch):
    """
    Обрабатывает батч для сегментации.
    """
    images, targets = batch
    
    # Определяем размер для ресайза
    target_size = (800, 800)
    
    # Преобразуем все изображения к одному размеру
    resized_images = []
    for i, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            image = torchvision.transforms.ToPILImage()(image)
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(target_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        resized_image = transform(image)
        resized_images.append(resized_image)
    
    images = torch.stack(resized_images).to(Config.DEVICE)
    
    # Обработка масок
    resized_masks = []
    for i, target_list in enumerate(targets):
        # Создаем пустую маску
        mask = torch.zeros(target_size, dtype=torch.float32)
        
        # Получаем размеры оригинального изображения
        if isinstance(images[i], torch.Tensor):
            orig_h, orig_w = images[i].shape[-2:]
        else:
            orig_h, orig_w = target_size
        
        # Масштабируем координаты
        scale_x = target_size[1] / orig_w
        scale_y = target_size[0] / orig_h
        
        # Обрабатываем каждую аннотацию в списке
        for target in target_list:
            # Проверяем наличие сегментации и bbox
            has_segmentation = 'segmentation' in target and target['segmentation']
            has_bbox = 'bbox' in target
            
            if has_segmentation:
                # Обрабатываем каждый полигон в сегментации
                for polygon in target['segmentation']:
                    # Масштабируем координаты
                    scaled_polygon = []
                    for j in range(0, len(polygon), 2):
                        x = int(polygon[j] * scale_x)
                        y = int(polygon[j+1] * scale_y)
                        scaled_polygon.extend([x, y])
                    
                    # Создаем маску из полигона
                    polygon_mask = create_mask_from_polygon(scaled_polygon, target_size)
                    # Объединяем с основной маской
                    mask = torch.logical_or(mask, polygon_mask).float()
            elif has_bbox:
                x, y, w, h = target['bbox']
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)
                
                # Проверяем границы
                x1 = max(0, min(x1, target_size[1]-1))
                y1 = max(0, min(y1, target_size[0]-1))
                x2 = max(0, min(x2, target_size[1]-1))
                y2 = max(0, min(y2, target_size[0]-1))
                
                # Заполняем маску
                mask[y1:y2, x1:x2] = 1.0
        
        resized_masks.append(mask)
    
    masks = torch.stack(resized_masks).to(Config.DEVICE)
    return images, masks

def calculate_segmentation_iou(pred_masks, target_masks):
    """Вычисляет IoU для сегментации"""
    intersection = torch.logical_and(pred_masks, target_masks).sum().float()
    union = torch.logical_or(pred_masks, target_masks).sum().float()
    return (intersection / union).item() if union > 0 else 0.0

def calculate_segmentation_accuracy(pred_masks, target_masks):
    """Вычисляет точность для сегментации"""
    correct = (pred_masks == target_masks).sum().float()
    total = target_masks.numel()
    return (correct / total).item()

def calculate_segmentation_precision(pred_masks, target_masks):
    """Вычисляет precision для сегментации"""
    true_positives = torch.logical_and(pred_masks, target_masks).sum().float()
    false_positives = torch.logical_and(pred_masks, torch.logical_not(target_masks)).sum().float()
    
    denominator = (true_positives + false_positives)
    if denominator > 0:
        return (true_positives / denominator).item()
    return 0.0

def calculate_segmentation_recall(pred_masks, target_masks):
    """Вычисляет recall для сегментации"""
    true_positives = torch.logical_and(pred_masks, target_masks).sum().float()
    false_negatives = torch.logical_and(torch.logical_not(pred_masks), target_masks).sum().float()
    return (true_positives / (true_positives + false_negatives)).item() if (true_positives + false_negatives) > 0 else 0.0

def calculate_segmentation_mean_iou(pred_masks, target_masks):
    """Вычисляет mean IoU для сегментации"""
    # Для каждого класса вычисляем IoU
    num_classes = max(pred_masks.max().item(), target_masks.max().item()) + 1
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred_masks == cls)
        target_cls = (target_masks == cls)
        
        intersection = torch.logical_and(pred_cls, target_cls).sum().float()
        union = torch.logical_or(pred_cls, target_cls).sum().float()
        
        if union > 0:
            ious.append((intersection / union).item())
    
    return sum(ious) / len(ious) if ious else 0.0

def calculate_maskrcnn_iou(predictions, target_masks):
    """Вычисляет IoU для Mask R-CNN"""
    total_iou = 0.0
    count = 0
    
    for pred, target in zip(predictions, target_masks):
        if len(pred['masks']) > 0 and len(target) > 0:
            pred_mask = pred['masks'][0, 0] > 0.5  # Берем первую маску
            intersection = torch.logical_and(pred_mask, target[0]).sum().float()
            union = torch.logical_or(pred_mask, target[0]).sum().float()
            if union > 0:
                total_iou += (intersection / union).item()
                count += 1
    
    return total_iou / count if count > 0 else 0.0

def calculate_maskrcnn_accuracy(predictions, target_masks):
    """Вычисляет точность для Mask R-CNN"""
    total_correct = 0
    total_pixels = 0
    
    for pred, target in zip(predictions, target_masks):
        if len(pred['masks']) > 0 and len(target) > 0:
            pred_mask = pred['masks'][0, 0] > 0.5  # Берем первую маску
            correct = (pred_mask == target[0]).sum().float()
            total_correct += correct.item()
            total_pixels += target[0].numel()
    
    return total_correct / total_pixels if total_pixels > 0 else 0.0

def calculate_maskrcnn_precision(predictions, target_masks):
    """Вычисляет precision для Mask R-CNN"""
    total_precision = 0.0
    count = 0
    
    for pred, target in zip(predictions, target_masks):
        if len(pred['masks']) > 0 and len(target) > 0:
            pred_mask = pred['masks'][0, 0] > 0.5  # Берем первую маску
            true_positives = torch.logical_and(pred_mask, target[0]).sum().float()
            false_positives = torch.logical_and(pred_mask, torch.logical_not(target[0])).sum().float()
            if (true_positives + false_positives) > 0:
                total_precision += (true_positives / (true_positives + false_positives)).item()
                count += 1
    
    return total_precision / count if count > 0 else 0.0

def calculate_maskrcnn_recall(predictions, target_masks):
    """Вычисляет recall для Mask R-CNN"""
    total_recall = 0.0
    count = 0
    
    for pred, target in zip(predictions, target_masks):
        if len(pred['masks']) > 0 and len(target) > 0:
            pred_mask = pred['masks'][0, 0] > 0.5  # Берем первую маску
            true_positives = torch.logical_and(pred_mask, target[0]).sum().float()
            false_negatives = torch.logical_and(torch.logical_not(pred_mask), target[0]).sum().float()
            if (true_positives + false_negatives) > 0:
                total_recall += (true_positives / (true_positives + false_negatives)).item()
                count += 1
    
    return total_recall / count if count > 0 else 0.0

def calculate_maskrcnn_mean_iou(predictions, target_masks):
    """Вычисляет mean IoU для Mask R-CNN"""
    total_iou = 0.0
    count = 0
    
    for pred, target in zip(predictions, target_masks):
        if len(pred['masks']) > 0 and len(target) > 0:
            pred_mask = pred['masks'][0, 0] > 0.5  # Берем первую маску
            intersection = torch.logical_and(pred_mask, target[0]).sum().float()
            union = torch.logical_or(pred_mask, target[0]).sum().float()
            if union > 0:
                total_iou += (intersection / union).item()
                count += 1
    
    return total_iou / count if count > 0 else 0.0

def calculate_segmentation_map(pred_masks, target_masks, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """Вычисляет mAP для сегментации"""
    aps = []
    
    # Преобразуем предсказания в вероятности (softmax)
    pred_probs = torch.nn.functional.softmax(pred_masks, dim=1)
    
    # Убедимся, что размерности совпадают
    if pred_probs.shape[2:] != target_masks.shape[1:]:
        pred_probs = torch.nn.functional.interpolate(pred_probs, size=target_masks.shape[1:], mode='bilinear', align_corners=False)
    
    # Для каждого порога
    for threshold in thresholds:
        # Применяем порог к вероятностям
        pred_binary = (pred_probs > threshold)
        target_binary = (target_masks > 0)
        
        # Убедимся, что размерности совпадают
        if pred_binary.shape != target_binary.shape:
            # Приводим к одинаковой размерности
            if len(pred_binary.shape) == 4 and len(target_binary.shape) == 3:
                # Если pred_binary имеет размерность [B, C, H, W], а target_binary [B, H, W]
                # Берем максимальное значение по каналам для pred_binary
                pred_binary = pred_binary.max(dim=1)[0]
            elif len(pred_binary.shape) == 3 and len(target_binary.shape) == 4:
                target_binary = target_binary.squeeze(1)
        
        # Вычисляем precision и recall для текущего порога
        true_positives = torch.logical_and(pred_binary, target_binary).sum().float()
        false_positives = torch.logical_and(pred_binary, torch.logical_not(target_binary)).sum().float()
        false_negatives = torch.logical_and(torch.logical_not(pred_binary), target_binary).sum().float()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # Вычисляем AP для текущего порога
        ap = precision * recall
        aps.append(float(ap))
    
    # Возвращаем среднее значение AP по всем порогам
    return sum(aps) / len(aps) if aps else 0.0

def calculate_maskrcnn_map(predictions, target_masks, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """Вычисляет mAP для Mask R-CNN"""
    total_map = 0.0
    count = 0
    
    for pred, target in zip(predictions, target_masks):
        if len(pred['masks']) > 0 and len(target) > 0:
            pred_mask = pred['masks'][0, 0]  # Берем первую маску
            target_mask = target[0]
            
            aps = []
            for threshold in thresholds:
                pred_binary = (pred_mask > threshold)
                target_binary = (target_mask > 0)
                
                # Вычисляем precision и recall для текущего порога
                true_positives = torch.logical_and(pred_binary, target_binary).sum().float()
                false_positives = torch.logical_and(pred_binary, torch.logical_not(target_binary)).sum().float()
                false_negatives = torch.logical_and(torch.logical_not(pred_binary), target_binary).sum().float()
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                
                # Вычисляем AP для текущего порога
                ap = precision * recall
                aps.append(ap)  # ap уже является скаляром
            
            if aps:
                total_map += sum(aps) / len(aps)
                count += 1
    
    return total_map / count if count > 0 else 0.0

def train_segmentation_step(engine, batch, model, optimizer, model_type):
    """
    Шаг обучения для модели сегментации.
    
    Args:
        engine: Ignite engine
        batch: Батч данных
        model: Модель сегментации
        optimizer: Оптимизатор
        model_type: Тип модели ('deeplabv3', 'fcn', 'unet' или 'maskrcnn')
    
    Returns:
        dict: Словарь с метриками
    """
    model.train()
    images, masks = process_segmentation_batch(batch)
    
    optimizer.zero_grad()
    
    # Для DeepLabV3 и FCN
    if model_type in ['deeplabv3', 'fcn']:
        outputs = model(images)['out']
        loss = torch.nn.functional.cross_entropy(outputs, masks.squeeze(1).long())
    # Для U-Net
    elif model_type == 'unet':
        outputs = model(images)
        # Для бинарной сегментации используем BCEWithLogitsLoss
        if outputs.shape[1] == 1:
            masks = masks.unsqueeze(1)  # Добавляем канал
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        else:
            loss = torch.nn.functional.cross_entropy(outputs, masks.squeeze(1).long())
    # Для Mask R-CNN
    else:
        loss_dict = model(images, [{'masks': mask} for mask in masks])
        loss = sum(loss for loss in loss_dict.values())
    
    loss.backward()
    optimizer.step()
    
    # Вычисляем метрики
    with torch.no_grad():
        if model_type == 'unet':
            if outputs.shape[1] == 1:
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            else:
                pred_masks = torch.argmax(outputs, dim=1)
        elif model_type in ['deeplabv3', 'fcn']:
            pred_masks = torch.argmax(outputs, dim=1)
        else:  # Mask R-CNN
            model.eval()
            predictions = model(images)
            iou = calculate_maskrcnn_iou(predictions, masks)
            mean_iou = calculate_maskrcnn_mean_iou(predictions, masks)
            accuracy = calculate_maskrcnn_accuracy(predictions, masks)
            precision = calculate_maskrcnn_precision(predictions, masks)
            recall = calculate_maskrcnn_recall(predictions, masks)
            map_score = calculate_maskrcnn_map(predictions, masks)
            return {
                'loss': loss.item(),
                'iou': iou,
                'mean_iou': mean_iou,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'map': map_score
            }
        
        target_masks = masks.squeeze(1)
        
        # Преобразуем в бинарные маски
        pred_binary = (pred_masks > 0).float()
        target_binary = (target_masks > 0).float()
        
        # Вычисляем метрики
        iou = calculate_segmentation_iou(pred_binary, target_binary)
        mean_iou = calculate_segmentation_mean_iou(pred_masks, target_masks)
        accuracy = calculate_segmentation_accuracy(pred_masks, target_masks)
        precision = calculate_segmentation_precision(pred_binary, target_binary)
        recall = calculate_segmentation_recall(pred_binary, target_binary)
        map_score = calculate_segmentation_map(outputs, target_masks)
    
    return {
        'loss': loss.item(),
        'iou': iou,
        'mean_iou': mean_iou,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'map': map_score
    }

def prepare_dataloader(data_root: str, ann_file: str, batch_size: int):
    """
    Подготавливает DataLoader для обучения.
    
    Args:
        data_root (str): Путь к директории с изображениями
        ann_file (str): Путь к файлу аннотаций
        batch_size (int): Размер батча
    
    Returns:
        DataLoader: Загрузчик данных
    """
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
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    return train_loader

def setup_metrics(trainer: Engine):
    """
    Настраивает метрики для обучения.
    
    Args:
        trainer (Engine): Ignite engine
    
    Returns:
        None
    """
    # Добавляем метрики
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x['iou']).attach(trainer, 'iou')
    RunningAverage(output_transform=lambda x: x['mean_iou']).attach(trainer, 'mean_iou')
    RunningAverage(output_transform=lambda x: x['accuracy']).attach(trainer, 'accuracy')
    RunningAverage(output_transform=lambda x: x['precision']).attach(trainer, 'precision')
    RunningAverage(output_transform=lambda x: x['recall']).attach(trainer, 'recall')
    RunningAverage(output_transform=lambda x: x['map']).attach(trainer, 'map')
    
    # Добавляем прогресс-бар
    ProgressBar(persist=True).attach(trainer, ['loss', 'iou', 'mean_iou', 'accuracy', 'precision', 'recall', 'map'])

def train_segmentation_model(
    model_type: str = 'deeplabv3',
    num_classes: int = None,
    num_epochs: int = Config.NUM_EPOCHS,
    learning_rate: float = 0.001,
    momentum: float = Config.MOMENTUM,
    weight_decay: float = Config.WEIGHT_DECAY,
    optimizer_type: str = 'adam',
    train_loader: DataLoader = None,
    device: torch.device = Config.DEVICE
):
    """
    Функция для запуска обучения модели сегментации.
    
    Args:
        model_type (str): Тип модели ('deeplabv3', 'fcn', 'unet' или 'maskrcnn')
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
    model = get_segmentation_model(model_type, num_classes)
    
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
    trainer = Engine(lambda engine, batch: train_segmentation_step(engine, batch, model, optimizer, model_type))
    
    # Настраиваем метрики
    setup_metrics(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        metrics = engine.state.metrics
        current_lr = optimizer.param_groups[0]['lr']
        
        # Обновляем историю
        history.update(metrics, current_lr)
        
        print(
            f"Epoch {engine.state.epoch}, "
            f"Loss: {metrics['loss']:.4f}, "
            f"IoU: {metrics['iou']:.4f}, "
            f"Mean IoU: {metrics['mean_iou']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
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

if __name__ == "__main__":
    # Пример использования для сегментации
    model, trainer, history = train_segmentation_model(
        model_type='unet',
        optimizer_type='adam',
        learning_rate=0.001
    ) 