from .rtsp import RTSPManager
from .storage import StorageManager
from .detection import DetectionService

rtsp_manager = RTSPManager()
storage_manager = StorageManager()
detection_service = DetectionService()

from .processor import Processor
processor = Processor()

__all__ = ['rtsp_manager', 'storage_manager', 'processor', 'detection_service']
