# YOFLO - Object detection and visual Q&A using Florence-2

__version__ = "1.4.0"

from .core import YOFLO
from .model import ModelManager
from .recording import RecordingManager
from .utils import ImageUtils, AlertLogger, get_youtube_stream_url, setup_logging
from .cli import main
from .client import YofloClient, detect, ask, is_ready

from .ptz import (
    PTZTracker,
    HIDPTZController,
    PTZController,
    ONVIFPTZController,
    HID_AVAILABLE,
    ONVIF_AVAILABLE,
)

__all__ = [
    'YOFLO',
    'ModelManager',
    'RecordingManager',
    'ImageUtils',
    'AlertLogger',
    'get_youtube_stream_url',
    'setup_logging',
    'main',
    'YofloClient',
    'detect',
    'ask',
    'is_ready',
    'PTZTracker',
    'HIDPTZController',
    'PTZController',
    'ONVIFPTZController',
    'HID_AVAILABLE',
    'ONVIF_AVAILABLE',
]
