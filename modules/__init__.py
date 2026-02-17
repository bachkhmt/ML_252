"""
ML Tabular Project - Python Modules
Học Máy (CO3001) - HK I (2025-2026)

Modules for data preprocessing, feature engineering, model training and evaluation.
"""

__version__ = "1.1.0"
__author__ = "Hoàng Xuân Bách, Nguyễn Việt Hùng, Trần Văn Hùng"

from . import config
from . import preprocessing
from . import features
from . import models
from . import evaluation

__all__ = [
    'config',
    'preprocessing',
    'features',
    'models',
    'evaluation'
]