"""
Model Training Module
Handles traditional ML and deep learning models
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # pip install xgboost
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None  # pip install lightgbm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from . import config


class ModelTrainer:
    """
    Model training and evaluation class
    """
    
    def __init__(self, verbose=True):
        """
        Initialize model trainer
        
        Parameters:
        -----------
        verbose : bool
            Print training progress
        """
        self.verbose = verbose
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def _print(self, message):
        """Print message if verbose mode is on"""
        if self.verbose:
            print(f"[Model Training] {message}")
    
    def get_traditional_models(self):
        """
        Get dictionary of traditional ML models
        
        Returns:
        --------
        dict
            Dictionary of model name to model instance
        """
        models_candidates = {
            'LogisticRegression': LogisticRegression(**config.MODELS_CONFIG['LogisticRegression']),
            'SVC': SVC(**config.MODELS_CONFIG['SVC']),
            'RandomForest': RandomForestClassifier(**config.MODELS_CONFIG['RandomForest']),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'DecisionTree': DecisionTreeClassifier(random_state=config.RANDOM_STATE),
            'GaussianNB': GaussianNB(),
        }
        if XGBClassifier is not None:
            models_candidates['XGBoost'] = XGBClassifier(**{k:v for k,v in config.MODELS_CONFIG['XGBoost'].items() if k != 'use_label_encoder'})
        if LGBMClassifier is not None:
            models_candidates['LightGBM'] = LGBMClassifier(**config.MODELS_CONFIG['LightGBM'])
        models = models_candidates
        
        return models
    
    def train_single_model(self, model, X_train, y_train, model_name='Model'):
        """
        Train a single model
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to train
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        model_name : str
            Name of the model
            
        Returns:
        --------
        model
            Trained model
        float
            Training time
        """
        self._print(f"Training {model_name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self._print(f"{model_name} trained in {training_time:.2f} seconds")
        
        return model, training_time
    
    def train_all_models(self, X_train, y_train):
        """
        Train all traditional ML models
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
            
        Returns:
        --------
        dict
            Dictionary of trained models and their training times
        """
        self._print("Training all traditional ML models...")
        
        models = self.get_traditional_models()
        results = {}
        
        for name, model in models.items():
            try:
                trained_model, training_time = self.train_single_model(
                    model, X_train, y_train, name
                )
                self.models[name] = trained_model
                results[name] = {'training_time': training_time}
            except Exception as e:
                self._print(f"Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
        """
        Perform cross-validation on a model
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to cross-validate
        X : array-like
            Features
        y : array-like
            Target
        cv : int
            Number of folds
        scoring : str
            Scoring metric
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
    
    def hyperparameter_tuning(self, model, X_train, y_train, param_grid, 
                             search_type='grid', cv=5, n_iter=10):
        """
        Perform hyperparameter tuning
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to tune
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        param_grid : dict
            Parameter grid for search
        search_type : str
            Type of search ('grid' or 'random')
        cv : int
            Number of cross-validation folds
        n_iter : int
            Number of iterations for random search
            
        Returns:
        --------
        estimator
            Best model found
        dict
            Best parameters
        float
            Best score
        """
        self._print(f"Performing {search_type} search for hyperparameter tuning...")
        
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv,
                scoring='accuracy', n_jobs=-1, verbose=1,
                random_state=config.RANDOM_STATE
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        search.fit(X_train, y_train)
        
        self._print(f"Best parameters: {search.best_params_}")
        self._print(f"Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def save_model(self, model, model_name, timestamp=None):
        """
        Save trained model to disk
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to save
        model_name : str
            Name of the model
        timestamp : str, optional
            Timestamp to append to filename
        """
        filepath = config.get_model_path(model_name, timestamp)
        joblib.dump(model, filepath)
        self._print(f"Model saved to {filepath}")
    
    def load_model(self, model_name, timestamp=None):
        """
        Load trained model from disk
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        timestamp : str, optional
            Timestamp of the saved model
            
        Returns:
        --------
        estimator
            Loaded model
        """
        filepath = config.get_model_path(model_name, timestamp)
        model = joblib.load(filepath)
        self._print(f"Model loaded from {filepath}")
        return model
    
    def get_model_info(self, model):
        """
        Get information about a trained model
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model
            
        Returns:
        --------
        dict
            Model information
        """
        info = {
            'model_type': type(model).__name__,
            'parameters': model.get_params()
        }
        
        # Get feature importances if available
        if hasattr(model, 'feature_importances_'):
            info['feature_importances'] = model.feature_importances_
        
        # Get coefficients if available
        if hasattr(model, 'coef_'):
            info['coefficients'] = model.coef_
        
        return info


class DeepLearningTrainer:
    """
    Deep learning model trainer (Bonus)
    """
    
    def __init__(self, model_type='MLP', verbose=True):
        """
        Initialize deep learning trainer
        
        Parameters:
        -----------
        model_type : str
            Type of model ('MLP' or 'TabNet')
        verbose : bool
            Print training progress
        """
        self.model_type = model_type
        self.verbose = verbose
        self.model = None
        self.history = None
        
    def _print(self, message):
        """Print message if verbose mode is on"""
        if self.verbose:
            print(f"[Deep Learning] {message}")
    
    def build_mlp(self, input_dim, output_dim, hidden_layers=None):
        """
        Build MLP model using PyTorch
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        output_dim : int
            Number of output classes
        hidden_layers : list
            List of hidden layer sizes
            
        Returns:
        --------
        torch.nn.Module
            MLP model
        """
        import torch
        import torch.nn as nn
        
        if hidden_layers is None:
            hidden_layers = config.DL_CONFIG['MLP']['hidden_layers']
        
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.DL_CONFIG['MLP']['dropout_rate']))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_dim))
        
        model = nn.Sequential(*layers)
        
        return model
    
    def train_mlp(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train MLP model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation target
            
        Returns:
        --------
        model
            Trained model
        dict
            Training history
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        self._print("Training MLP model...")
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.DL_CONFIG['MLP']['batch_size'],
            shuffle=True
        )
        
        # Build model
        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y_train))
        
        model = self.build_mlp(input_dim, output_dim)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.DL_CONFIG['MLP']['learning_rate']
        )
        
        # Training loop
        history = {'train_loss': [], 'train_acc': []}
        
        for epoch in range(config.DL_CONFIG['MLP']['epochs']):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            epoch_loss = train_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            if (epoch + 1) % 10 == 0:
                self._print(f"Epoch {epoch+1}/{config.DL_CONFIG['MLP']['epochs']} - "
                          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
        self.model = model
        self.history = history
        
        return model, history
    
    def train_tabnet(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train TabNet model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation target
            
        Returns:
        --------
        model
            Trained model
        """
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError:
            self._print("pytorch-tabnet not installed. Install with: pip install pytorch-tabnet")
            return None
        
        self._print("Training TabNet model...")
        
        model = TabNetClassifier(**config.DL_CONFIG['TabNet'])
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        else:
            eval_set = None
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=['accuracy']
        )
        
        self.model = model
        
        return model