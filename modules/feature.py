"""
Feature Engineering Module
Handles feature selection, extraction, and transformation
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from . import config


class FeatureEngineer:
    """
    Feature engineering class for:
    - Feature selection
    - Dimensionality reduction
    - Feature creation
    """
    
    def __init__(self, verbose=True):
        """
        Initialize feature engineer
        
        Parameters:
        -----------
        verbose : bool
            Print processing steps
        """
        self.verbose = verbose
        self.selected_features = None
        self.pca = None
        self.feature_importances = None
        
    def _print(self, message):
        """Print message if verbose mode is on"""
        if self.verbose:
            print(f"[Feature Engineering] {message}")
    
    def remove_correlated_features(self, X, threshold=0.95):
        """
        Remove highly correlated features
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        threshold : float
            Correlation threshold (default: 0.95)
            
        Returns:
        --------
        pd.DataFrame
            Features with correlations removed
        list
            Removed feature names
        """
        self._print(f"Removing features with correlation > {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        self._print(f"Removing {len(to_drop)} highly correlated features")
        
        return X.drop(columns=to_drop), to_drop
    
    def select_k_best(self, X, y, k=10, score_func='f_classif'):
        """
        Select K best features using statistical tests
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
        k : int
            Number of features to select
        score_func : str
            Scoring function ('f_classif' or 'mutual_info')
            
        Returns:
        --------
        pd.DataFrame
            Selected features
        list
            Selected feature names
        """
        self._print(f"Selecting {k} best features using {score_func}")
        
        if score_func == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif score_func == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown score function: {score_func}")
        
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        self._print(f"Selected features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def recursive_feature_elimination(self, X, y, n_features=10, estimator=None):
        """
        Perform Recursive Feature Elimination
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
        n_features : int
            Number of features to select
        estimator : sklearn estimator
            Base estimator for RFE (default: RandomForest)
            
        Returns:
        --------
        pd.DataFrame
            Selected features
        list
            Selected feature names
        """
        self._print(f"Performing RFE to select {n_features} features")
        
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        self._print(f"Selected features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def feature_importance_selection(self, X, y, threshold='median', estimator=None):
        """
        Select features based on feature importance
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
        threshold : str or float
            Threshold for selection ('mean', 'median', or float value)
        estimator : sklearn estimator
            Base estimator (default: RandomForest)
            
        Returns:
        --------
        pd.DataFrame
            Selected features
        dict
            Feature importances
        """
        self._print("Selecting features based on importance")
        
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        
        selector = SelectFromModel(estimator, threshold=threshold, prefit=False)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature importances
        estimator.fit(X, y)
        importances = dict(zip(X.columns, estimator.feature_importances_))
        self.feature_importances = importances
        
        self._print(f"Selected {len(selected_features)} features")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), importances
    
    def apply_pca(self, X, variance_ratio=0.95, fit=True):
        """
        Apply PCA for dimensionality reduction
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Input features
        variance_ratio : float
            Proportion of variance to retain (0 to 1)
        fit : bool
            Whether to fit PCA or just transform
            
        Returns:
        --------
        np.ndarray
            Transformed features
        int
            Number of components selected
        """
        if fit:
            self._print(f"Applying PCA with {variance_ratio*100}% variance retention")
            self.pca = PCA(n_components=variance_ratio, random_state=config.RANDOM_STATE)
            X_pca = self.pca.fit_transform(X)
            n_components = self.pca.n_components_
            self._print(f"Reduced from {X.shape[1]} to {n_components} components")
        else:
            X_pca = self.pca.transform(X)
            n_components = self.pca.n_components_
        
        return X_pca, n_components
    
    def create_polynomial_features(self, X, degree=2):
        """
        Create polynomial features
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        degree : int
            Degree of polynomial features
            
        Returns:
        --------
        pd.DataFrame
            Features with polynomial terms
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        self._print(f"Creating polynomial features of degree {degree}")
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        feature_names = poly.get_feature_names_out(X.columns)
        
        return pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    def create_interaction_features(self, X, features_to_interact):
        """
        Create interaction features between specified columns
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        features_to_interact : list of tuples
            Pairs of features to create interactions
            Example: [('age', 'income'), ('height', 'weight')]
            
        Returns:
        --------
        pd.DataFrame
            Original features plus interaction terms
        """
        self._print(f"Creating {len(features_to_interact)} interaction features")
        
        X_copy = X.copy()
        
        for feat1, feat2 in features_to_interact:
            interaction_name = f"{feat1}_x_{feat2}"
            X_copy[interaction_name] = X_copy[feat1] * X_copy[feat2]
            self._print(f"Created: {interaction_name}")
        
        return X_copy
    
    def bin_numeric_features(self, X, features_to_bin, n_bins=5, strategy='quantile'):
        """
        Bin numeric features into categories
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        features_to_bin : list
            List of feature names to bin
        n_bins : int
            Number of bins
        strategy : str
            Binning strategy ('uniform', 'quantile', 'kmeans')
            
        Returns:
        --------
        pd.DataFrame
            Features with binned columns
        """
        from sklearn.preprocessing import KBinsDiscretizer
        
        self._print(f"Binning {len(features_to_bin)} features into {n_bins} bins")
        
        X_copy = X.copy()
        
        binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        
        for feature in features_to_bin:
            binned_name = f"{feature}_binned"
            X_copy[binned_name] = binner.fit_transform(X_copy[[feature]])
        
        return X_copy


def get_feature_statistics(X, y=None):
    """
    Get comprehensive statistics about features
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : pd.Series, optional
        Target variable
        
    Returns:
    --------
    dict
        Dictionary containing various feature statistics
    """
    stats = {
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'feature_names': X.columns.tolist(),
        'numeric_features': X.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features': X.select_dtypes(include=['object', 'category']).columns.tolist(),
        'missing_values': X.isnull().sum().to_dict(),
        'data_types': X.dtypes.to_dict()
    }
    
    # Correlation with target if provided
    if y is not None:
        correlations = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            correlations[col] = np.corrcoef(X[col], y)[0, 1]
        stats['target_correlation'] = correlations
    
    return stats


def save_features(X, y, feature_names, prefix=''):
    """
    Save processed features to disk
    
    Parameters:
    -----------
    X : np.ndarray or pd.DataFrame
        Features to save
    y : np.ndarray or pd.Series
        Target variable to save
    feature_names : list
        List of feature names
    prefix : str
        Prefix for filename (e.g., 'train_', 'test_')
    """
    np.save(config.FEATURES_DIR / f'{prefix}X.npy', X)
    np.save(config.FEATURES_DIR / f'{prefix}y.npy', y)
    
    with open(config.FEATURES_DIR / f'{prefix}feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    print(f"Features saved to {config.FEATURES_DIR}")


def load_features(prefix=''):
    """
    Load processed features from disk
    
    Parameters:
    -----------
    prefix : str
        Prefix for filename (e.g., 'train_', 'test_')
        
    Returns:
    --------
    X, y, feature_names : tuple
        Loaded features, target, and feature names
    """
    X = np.load(config.FEATURES_DIR / f'{prefix}X.npy')
    y = np.load(config.FEATURES_DIR / f'{prefix}y.npy')
    
    with open(config.FEATURES_DIR / f'{prefix}feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"Features loaded from {config.FEATURES_DIR}")
    
    return X, y, feature_names