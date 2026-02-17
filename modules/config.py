"""
Configuration file for ML Tabular Project
Contains all hyperparameters and settings for the pipeline
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DIR = PROJECT_ROOT / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 MODELS_DIR, FEATURES_DIR, REPORTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # Validation split from training data

# Feature engineering configuration
CORRELATION_THRESHOLD = 0.95  # Remove highly correlated features
PCA_VARIANCE_RATIOS = [0.90, 0.95, 0.99]  # Try different variance retention

# Preprocessing configuration
IMPUTATION_METHODS = {
    'numeric': ['mean', 'median', 'knn'],
    'categorical': ['mode', 'constant']
}

ENCODING_METHODS = {
    'onehot': ['OneHotEncoder'],
    'label': ['LabelEncoder'],
    'target': ['TargetEncoder']
}

SCALING_METHODS = ['standard', 'minmax', 'robust']

# KNN Imputer configuration
KNN_N_NEIGHBORS = 5

# Model configurations
MODELS_CONFIG = {
    'LogisticRegression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced'
    },
    'SVC': {
        'random_state': RANDOM_STATE,
        'probability': True,
        'class_weight': 'balanced'
    },
    'RandomForest': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'class_weight': 'balanced'
    },
    'XGBoost': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    },
    'LightGBM': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'verbose': -1
    }
}

# Hyperparameter tuning configuration
PARAM_GRIDS = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
}

# Cross-validation configuration
CV_FOLDS = 5

# Deep Learning configuration (Bonus)
DL_CONFIG = {
    'MLP': {
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 10
    },
    'TabNet': {
        'n_d': 8,
        'n_a': 8,
        'n_steps': 3,
        'gamma': 1.3,
        'n_independent': 2,
        'n_shared': 2,
        'lambda_sparse': 1e-3,
        'optimizer_params': {'lr': 2e-2},
        'scheduler_params': {'step_size': 10, 'gamma': 0.9},
        'max_epochs': 100,
        'patience': 10,
        'batch_size': 1024
    }
}

# Visualization configuration
FIGURE_SIZE = (12, 8)
FIGURE_DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Evaluation metrics
CLASSIFICATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc'
]

# Logging configuration
VERBOSE = True
LOG_LEVEL = 'INFO'

# File naming conventions
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

def get_model_path(model_name, timestamp=None):
    """Generate model save path"""
    if timestamp:
        return MODELS_DIR / f"{model_name}_{timestamp}.pkl"
    return MODELS_DIR / f"{model_name}.pkl"

def get_feature_path(feature_name, timestamp=None):
    """Generate feature save path"""
    if timestamp:
        return FEATURES_DIR / f"{feature_name}_{timestamp}.npy"
    return FEATURES_DIR / f"{feature_name}.npy"

def get_figure_path(figure_name):
    """Generate figure save path"""
    return FIGURES_DIR / f"{figure_name}.png"