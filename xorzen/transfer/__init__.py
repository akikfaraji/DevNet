"""
XORZENX Transfer - Now with ANY model support
"""

from .transfer import TeacherExtractor, XORZENXTransferLearning, quick_transfer
from .universal import UniversalTransfer, list_recommended_models, RECOMMENDED_MODELS

__all__ = [
    'TeacherExtractor',
    'XORZENXTransferLearning',
    'quick_transfer',
    'UniversalTransfer',
    'list_recommended_models',
    'RECOMMENDED_MODELS',
]
