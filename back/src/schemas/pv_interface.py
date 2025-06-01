from enum import IntEnum
from typing import Dict, List, Union


class State(IntEnum):
    UNKNOWN: int = 0
    RUNNING: int = 1
    FAILED: int = 2


class Action(IntEnum):
    START: int = 0
    STOP: int = 1
    RESTART: int = 2


class Info:
    """
    Fields:
     - `state`: Current state of the process.
     - `start`: Time of the initial process launch.
     - `rtsp_start`: Time of the last connection to RTSP source.
     - `output_switch`: Wether process is publishing RTMP.
    """

    def __init__(self, state: State, start: float, rtsp_start: float, output_switch: bool):
        self.state: State = state
        self.start: float = start
        self.rtsp_start: float = rtsp_start
        self.output_switch: bool = output_switch

    def __str__(self):
        return '<Info: state={}, start={}, rtsp_start={}, output_switch={}>'.format(self.state, self.start, self.rtsp_start, self.output_switch)


class Arguments:
    """
    Necessary args for Core process to open RTSP stream (create GStreamer pipe).

    `location` should be `rstp://<host>:<port>/<path>`.
    If RTSP source does not have `user` or `password` they should be set to None.
    """

    def __init__(self, location: str, user: str = None, password: str = None):
        self.location: str = location
        self.user: str = user
        self.password: str = password

    def __str__(self):
        return '<Arguments: location={}, user={}, password={}>'.format(self.location, self.user, self.password)


# -----------------------------
#   FrameInfo
# -----------------------------

class FrameInfo:
    """
    Analytical information of a single frame. Frames should be matched using `source_id` and `datetime`.
    All lists have the same length (number of people).

    Note: `coords` is a nested list: [[x1, y1, x2, y2], [x1, y1, x2, y2]... [x1, y1, x2, y2]]
    """

    def __init__(self, source_id, session_uuid, datetime, coords, tracker_ids, masks, helmets, vests, social_distansing):
        self.source_id: str = source_id
        self.session_uuid: str = session_uuid
        self.datetime: float = datetime
        self.coords: list = coords
        self.tracked_ids: List[int] = tracker_ids
        self.masks: List[bool] = masks
        self.helmets: List[bool] = helmets
        self.vests: List[bool] = vests
        self.social_distansing: List[bool] = social_distansing


# -----------------------------
#   Base Message
# -----------------------------

class RemoteMessage:
    """ Base class for Kafka messages. """

    def __init__(self, cmd_id: str):
        self.cmd_id: str = cmd_id

# -----------------------------
#   Remote Commands
# -----------------------------


class PsCommand(RemoteMessage):
    """
    Process status commands.
    Responce will contain `dict` in value filed.

    Dict:
    Key - `source_id` of the proccess.
    Value - `Info` object.
    """

    def __init__(self, cmd_id: str, source_id: str = None):
        super(PsCommand, self).__init__(cmd_id=cmd_id)
        self.source_id = source_id

    def __str__(self):
        return '<PsCommand {}: source_id={}>'.format(self.cmd_id, self.source_id)


class OutputSwitchCommand(RemoteMessage):
    """ Switches RTMP output ON/OFF for specified `source_id`. """

    def __init__(self, cmd_id: str, source_id: str, switch: bool):
        super(OutputSwitchCommand, self).__init__(cmd_id=cmd_id)
        self.source_id = source_id
        self.switch = switch

    def __str__(self):
        return '<OutputSwitchCommand {}: source_id={}, switch={}>'.format(self.cmd_id, self.source_id, self.switch)


class ActionCommand(RemoteMessage):
    """
    Sends a command to mutate processes state.

    If `Action` is:
     - `START`: Core starts new process with `source_id`. `Arguments` are passed to process.
     - `STOP`: Core stops process with `source_id` and releases `source_id`. `Arguments` are ignored.
     - `RESTART`: Core stops and starts process with `source_id`.
        If `Arguments` are not None new ones are used, otherwise old `Arguments` are used.

    Note: Then process starts `output_switch` defaults to OFF. If RTMP is needed send `OutputSwitchCommand` after Restart.
    """

    def __init__(self, cmd_id: str, source_id: str, action: Action, args: Arguments = None):
        super(ActionCommand, self).__init__(cmd_id=cmd_id)
        self.source_id: str = source_id
        self.action: Action = action
        self.args: Arguments = args

    def __str__(self):
        return '<ActionCommand {}: source_id={}, action={}, args={}>'.format(self.cmd_id, self.source_id, self.action, self.args)

# -----------------------------
#   Response
# -----------------------------


class Response(RemoteMessage):
    """
    Tells the result of the command.
    `cmd_id` should be used to match responce with the command.
    """

    def __init__(self, cmd_id: str, successful: bool, message: str, value: Union[Dict[str, Info], None]):
        super(Response, self).__init__(cmd_id=cmd_id)
        self.successful: bool = successful
        self.message: str = message
        self.value: Union[Dict[str, Info], None] = value

    def __str__(self):
        if self.value is not None:
            value_str = '"' + ', '.join('{}={}'.format(k, v) for (k, v) in self.value.items()) + '"'
        else:
            value_str = str(self.value)
        return '<Response {}: successful={}, message={}, value={}>'.format(self.cmd_id, self.successful, self.message, value_str)