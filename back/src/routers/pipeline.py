from fastapi import APIRouter, status, Depends, HTTPException

from schemas.pipeline import StartingCam, ActionResponse
from schemas.pv_interface import Action
from db import get_database, Session
from broker.kafka import kafkaManager, ActionError

router = APIRouter()

def cs_processing(*args, cs, source, action):
    try:
        kafkaManager.action(cs.location, cs.login, cs.password, source, action)
        resp = ActionResponse(source_id=source, status=True)
    except ActionError as er:
        resp = ActionResponse(source_id=source, status=False, message=str(er))
    return resp


@router.post('/start')
def start_cam(
    cam_info: StartingCam,
    db: Session = Depends(get_database),
):
    resp = cs_processing(cs=cam_info, source='source', action=Action.START)
    if resp.status is False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)


@router.post('/stop')
def stop_cam(
    cam_info: StartingCam,
    db: Session = Depends(get_database),
):
    resp = cs_processing(cs=cam_info, source='source', action=Action.STOP)
    if resp.status is False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
