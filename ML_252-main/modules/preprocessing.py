"""
Data Preprocessing Module
Handles missing values, encoding, and scaling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

from . import config

# Optional: category_encoders for TargetEncoder
try:
    from category_encoders import TargetEncoder as _TargetEncoder
    _HAS_CE = True
except ImportError:
    _HAS_CE = False


class DataPreprocessor:
    """
    Main preprocessing class that handles:
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    """
    
    def __init__(self, imputation_method='mean', encoding_method='onehot',
                 scaling_method='standard', verbose=True):
        self.imputation_method = imputation_method
        self.encoding_method   = encoding_method
        self.scaling_method    = scaling_method
        self.verbose           = verbose

        self.imputer           = None
        self.encoder           = None
        self.scaler            = None
        self.numeric_columns   = None
        self.categorical_columns = None
        self.feature_names     = None

    def _print(self, message):
        if self.verbose:
            print(f"[Preprocessing] {message}")

    def identify_column_types(self, df):
        numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self._print(f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns")
        return numeric_cols, categorical_cols

    def handle_missing_values(self, df, fit=True):
        if df.isnull().sum().sum() == 0:
            return df
        self._print(f"Imputing missing values using {self.imputation_method}")
        df_copy = df.copy()
        if self.numeric_columns:
            if self.imputation_method == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif self.imputation_method == 'median':
                imputer = SimpleImputer(strategy='median')
            elif self.imputation_method == 'knn':
                imputer = KNNImputer(n_neighbors=config.KNN_N_NEIGHBORS)
            else:
                imputer = SimpleImputer(strategy='mean')
            if fit:
                self.imputer = imputer
                df_copy[self.numeric_columns] = self.imputer.fit_transform(df_copy[self.numeric_columns])
            else:
                df_copy[self.numeric_columns] = self.imputer.transform(df_copy[self.numeric_columns])
        if self.categorical_columns:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            if fit:
                df_copy[self.categorical_columns] = cat_imputer.fit_transform(df_copy[self.categorical_columns])
            else:
                df_copy[self.categorical_columns] = cat_imputer.transform(df_copy[self.categorical_columns])
        return df_copy

    def encode_categorical(self, df, target=None, fit=True):
        if not self.categorical_columns:
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
                    index=df_copy.index)
            else:
                encoded = self.encoder.transform(df_copy[self.categorical_columns])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=self.encoder.get_feature_names_out(self.categorical_columns),
                    index=df_copy.index)
            df_copy = pd.concat([df_copy.drop(columns=self.categorical_columns), encoded_df], axis=1)
        elif self.encoding_method == 'label':
            if not hasattr(self, '_label_encoders'):
                self._label_encoders = {}
            for col in self.categorical_columns:
                if fit:
                    le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    df_copy[[col]] = le.fit_transform(df_copy[[col]].astype(str))
                    self._label_encoders[col] = le
                else:
                    le = self._label_encoders[col]
                    df_copy[[col]] = le.transform(df_copy[[col]].astype(str))
        elif self.encoding_method == 'target':
            if not _HAS_CE:
                self._print("category_encoders not installed — falling back to OneHot")
                self.encoding_method = 'onehot'
                return self.encode_categorical(df, target, fit)
            if target is None and fit:
                raise ValueError("Target variable required for target encoding")
            if fit:
                self.encoder = _TargetEncoder(cols=self.categorical_columns)
                df_copy[self.categorical_columns] = self.encoder.fit_transform(
                    df_copy[self.categorical_columns], target)
            else:
                df_copy[self.categorical_columns] = self.encoder.transform(
                    df_copy[self.categorical_columns])
        return df_copy

    def scale_features(self, df, fit=True):
        self._print(f"Scaling features using {self.scaling_method}")
        scalers = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}
        if self.scaling_method not in scalers:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        if fit:
            self.scaler = scalers[self.scaling_method]
            scaled_data = self.scaler.fit_transform(df)
        else:
            scaled_data = self.scaler.transform(df)
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    def fit_transform(self, X, y=None):
        self._print("Starting preprocessing pipeline (fit_transform)...")
        self.numeric_columns, self.categorical_columns = self.identify_column_types(X)
        X_imputed  = self.handle_missing_values(X, fit=True)
        X_encoded  = self.encode_categorical(X_imputed, target=y, fit=True)
        X_scaled   = self.scale_features(X_encoded, fit=True)
        self.feature_names = X_scaled.columns.tolist()
        self._print(f"Preprocessing complete. Final shape: {X_scaled.shape}")
        return X_scaled

    def transform(self, X):
        self._print("Transforming data using fitted preprocessor...")
        X_imputed = self.handle_missing_values(X, fit=False)
        X_encoded = self.encode_categorical(X_imputed, fit=False)
        X_scaled  = self.scale_features(X_encoded, fit=False)
        return X_scaled


def check_missing_values(df):
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_Percentage': missing_pct})
    return missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)


def remove_duplicates(df, verbose=True):
    n_dup = df.duplicated().sum()
    if verbose:
        print(f"Found {n_dup} duplicate rows")
    if n_dup > 0:
        df_clean = df.drop_duplicates()
        if verbose:
            print(f"Removed {n_dup} duplicates. New shape: {df_clean.shape}")
        return df_clean
    return df


def handle_outliers(df, columns=None, method='iqr', threshold=1.5):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    df_clean = df.copy()
    outlier_info = {}
    for col in columns:
        if method == 'iqr':
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - threshold * IQR, Q3 + threshold * IQR
            outlier_info[col] = len(df[(df[col] < lb) | (df[col] > ub)])
            df_clean[col] = df_clean[col].clip(lb, ub)
        elif method == 'zscore':
            from scipy import stats
            z = np.abs(stats.zscore(df[col]))
            outlier_info[col] = len(df[z > threshold])
            df_clean = df_clean[z <= threshold]
    return df_clean, outlier_info