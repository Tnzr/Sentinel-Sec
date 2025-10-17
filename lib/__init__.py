"""
Face Recognition System Library
Contains core functionality for face detection, recognition, and database_db management
"""

__version__ = "1.0.0"
__author__ = "Python Conference Panama"

from .database import DatabaseManager
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .face_analyzer import FaceAnalyzer
from .utils import *

__all__ = [
    'DatabaseManager',
    'FaceDetector',
    'FaceRecognizer',
    'FaceAnalyzer'
]