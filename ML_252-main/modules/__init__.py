"""
ML Tabular Project - Python Modules
Học Máy (CO3001) - HK I (2025-2026)

Modules for data preprocessing, feature engineering, model training and evaluation.
"""

__version__ = "1.1.0"
__author__ = "Hoàng Xuân Bách, Nguyễn Việt Hùng"

from . import config
from . import preprocessing
from . import feature          # file tên feature.py
from . import models
from . import evaluate         # file tên evaluate.py

__all__ = ["config", "preprocessing", "feature", "models", "evaluate"]
