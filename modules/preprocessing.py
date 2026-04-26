"""
Data Preprocessing Module
Handles missing values, encoding, and scaling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

from . import config


class DataPreprocessor:
    """
    Main preprocessing class that handles:
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    """
    
    def __init__(self, imputation_method='mean', encoding_method='onehot', 
                 scaling_method='standard', verbose=True):
        """
        Initialize preprocessor
        
        Parameters:
        -----------
        imputation_method : str
            Method for handling missing values ('mean', 'median', 'mode', 'knn')
        encoding_method : str
            Method for encoding categorical variables ('onehot', 'label', 'target')
        scaling_method : str
            Method for scaling features ('standard', 'minmax', 'robust')
        verbose : bool
            Print processing steps
        """
        self.imputation_method = imputation_method
        self.encoding_method = encoding_method
        self.scaling_method = scaling_method
        self.verbose = verbose
        
        # Initialize transformers
        self.imputer = None
        self.encoder = None
        self.scaler = None
        
        # Store column information
        self.numeric_columns = None
        self.categorical_columns = None
        self.feature_names = None
        
    def _print(self, message):
        """Print message if verbose mode is on"""
        if self.verbose:
            print(f"[Preprocessing] {message}")
    
    def identify_column_types(self, df):
        """
        Identify numeric and categorical columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        numeric_cols, categorical_cols : lists
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self._print(f"Found {len(numeric_cols)} numeric columns")
        self._print(f"Found {len(categorical_cols)} categorical columns")
        
        return numeric_cols, categorical_cols
    
    def handle_missing_values(self, df, fit=True):
        """
        Handle missing values using specified imputation method
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            Whether to fit the imputer or just transform
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with imputed values
        """
        if df.isnull().sum().sum() == 0:
            self._print("No missing values found")
            return df
        
        self._print(f"Imputing missing values using {self.imputation_method} method")
        
        df_copy = df.copy()
        
        # Handle numeric columns
        if self.numeric_columns:
            if self.imputation_method == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif self.imputation_method == 'median':
                imputer = SimpleImputer(strategy='median')
            elif self.imputation_method == 'knn':
                imputer = KNNImputer(n_neighbors=config.KNN_N_NEIGHBORS)
            
            if fit:
                self.imputer = imputer
                df_copy[self.numeric_columns] = self.imputer.fit_transform(df_copy[self.numeric_columns])
            else:
                df_copy[self.numeric_columns] = self.imputer.transform(df_copy[self.numeric_columns])
        
        # Handle categorical columns
        if self.categorical_columns:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            if fit:
                df_copy[self.categorical_columns] = cat_imputer.fit_transform(
                    df_copy[self.categorical_columns]
                )
            else:
                df_copy[self.categorical_columns] = cat_imputer.transform(
                    df_copy[self.categorical_columns]
                )
        
        return df_copy
    
    def encode_categorical(self, df, target=None, fit=True):
        """
        Encode categorical variables
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target : pd.Series
            Target variable (required for target encoding)
        fit : bool
            Whether to fit the encoder or just transform
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with encoded categorical variables
        """
        if not self.categorical_columns:
            self._print("No categorical columns to encode")
            return df
        
        self._print(f"Encoding categorical variables using {self.encoding_method}")
        
        df_copy = df.copy()
        
        if self.encoding_method == 'onehot':
            if fit:
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = self.encoder.fit_transform(df_copy[self.categorical_columns])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=self.encoder.get_feature_names_out(self.categorical_columns),
                    index=df_copy.index
                )
            else:
                encoded = self.encoder.transform(df_copy[self.categorical_columns])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=self.encoder.get_feature_names_out(self.categorical_columns),
                    index=df_copy.index
                )
            
            # Drop original categorical columns and add encoded ones
            df_copy = df_copy.drop(columns=self.categorical_columns)
            df_copy = pd.concat([df_copy, encoded_df], axis=1)
            
        elif self.encoding_method == 'label':
            for col in self.categorical_columns:
                le = LabelEncoder()
                if fit:
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                else:
                    df_copy[col] = le.transform(df_copy[col].astype(str))
                    
        elif self.encoding_method == 'target':
            if target is None and fit:
                raise ValueError("Target variable required for target encoding")
            
            if fit:
                self.encoder = TargetEncoder(cols=self.categorical_columns)
                df_copy[self.categorical_columns] = self.encoder.fit_transform(
                    df_copy[self.categorical_columns], target
                )
            else:
                df_copy[self.categorical_columns] = self.encoder.transform(
                    df_copy[self.categorical_columns]
                )
        
        return df_copy
    
    def scale_features(self, df, fit=True):
        """
        Scale features using specified method
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            Whether to fit the scaler or just transform
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with scaled features
        """
        self._print(f"Scaling features using {self.scaling_method}")
        
        if self.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        if fit:
            self.scaler = scaler
            scaled_data = self.scaler.fit_transform(df)
        else:
            scaled_data = self.scaler.transform(df)
        
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform the data through the full pipeline
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series, optional
            Target variable (required for target encoding)
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed features
        """
        self._print("Starting preprocessing pipeline (fit_transform)...")
        
        # Identify column types
        self.numeric_columns, self.categorical_columns = self.identify_column_types(X)
        
        # Step 1: Handle missing values
        X_imputed = self.handle_missing_values(X, fit=True)
        
        # Step 2: Encode categorical variables
        X_encoded = self.encode_categorical(X_imputed, target=y, fit=True)
        
        # Step 3: Scale features
        X_scaled = self.scale_features(X_encoded, fit=True)
        
        # Store feature names
        self.feature_names = X_scaled.columns.tolist()
        
        self._print(f"Preprocessing complete. Final shape: {X_scaled.shape}")
        
        return X_scaled
    
    def transform(self, X):
        """
        Transform new data using fitted preprocessor
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed features
        """
        self._print("Transforming data using fitted preprocessor...")
        
        # Step 1: Handle missing values
        X_imputed = self.handle_missing_values(X, fit=False)
        
        # Step 2: Encode categorical variables
        X_encoded = self.encode_categorical(X_imputed, fit=False)
        
        # Step 3: Scale features
        X_scaled = self.scale_features(X_encoded, fit=False)
        
        return X_scaled


def check_missing_values(df):
    """
    Check and report missing values in dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Summary of missing values
    """
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    return missing_df


def remove_duplicates(df, verbose=True):
    """
    Remove duplicate rows from dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Print information about duplicates
        
    Returns:
    --------
    pd.DataFrame
        Dataframe without duplicates
    """
    n_duplicates = df.duplicated().sum()
    
    if verbose:
        print(f"Found {n_duplicates} duplicate rows")
    
    if n_duplicates > 0:
        df_clean = df.drop_duplicates()
        if verbose:
            print(f"Removed {n_duplicates} duplicates. New shape: {df_clean.shape}")
        return df_clean
    
    return df


def handle_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Detect and handle outliers using IQR method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to check for outliers (default: all numeric columns)
    method : str
        Method to use ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection (1.5 for IQR, 3 for z-score)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers handled
    dict
        Information about outliers found
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    outlier_info = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            # Cap outliers at bounds
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = df[z_scores > threshold]
            
            # Remove outliers
            df_clean = df_clean[z_scores <= threshold]
        
        outlier_info[col] = len(outliers)
    
    return df_clean, outlier_info