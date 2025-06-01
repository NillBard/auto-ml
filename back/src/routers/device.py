import torch
import psutil

from fastapi import APIRouter

from schemas.device import Device

router = APIRouter()


@router.get("/gpu")
async def get_gpu_memory():
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 1024 ** 3
        total = torch.cuda.mem_get_info()[1] / 1024 ** 3
        return Device(
            total=f'{total:.2f} Гб',
            usage=f'{total - free:.2f} Гб'
        )
    raise


@router.get("/cpu")
async def get_cpu_usage():
    return Device(
        total=f'{psutil.virtual_memory() [0] / 1024 ** 3:.2f} Гб',
        usage=f'{psutil.virtual_memory() [3] / 1024 ** 3:.2f} Гб'
    )
