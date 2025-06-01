from fastapi import APIRouter

from routers.pipeline import router as pipeline
from routers.train import router as train
from routers.login import router as login
from routers.dataset import router as dataset
from routers.device import router as device
from routers.processing import router as processing

router = APIRouter()

router.include_router(pipeline, prefix='/api/pipeline', tags=['Pipeline'])
router.include_router(train, prefix='/api/train', tags=['Train'])
router.include_router(login, prefix='/api/user', tags=['Login'])
router.include_router(dataset, prefix='/api/dataset', tags=['Dataset'])
router.include_router(device, prefix='/api/device', tags=['Device'])
router.include_router(processing, prefix='/api/processing', tags=['Processing'])
