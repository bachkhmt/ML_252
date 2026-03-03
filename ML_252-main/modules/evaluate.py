"""
Model Evaluation Module
Handles metrics calculation and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, auc, precision_recall_curve
)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

from . import config


class ModelEvaluator:
    """
    Model evaluation and visualization class
    """
    
    def __init__(self, verbose=True):
        """
        Initialize model evaluator
        
        Parameters:
        -----------
        verbose : bool
            Print evaluation results
        """
        self.verbose = verbose
        self.results = {}
        
    def _print(self, message):
        """Print message if verbose mode is on"""
        if self.verbose:
            print(f"[Evaluation] {message}")
    
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """
        Evaluate a single model on test data
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Dictionary containing all evaluation metrics
        """
        self._print(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC-AUC if model has predict_proba
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba, 
                                                       multi_class='ovr', average='weighted')
            except Exception as e:
                self._print(f"Could not calculate ROC-AUC: {str(e)}")
        
        # Store results
        self.results[model_name] = metrics
        
        # Print results
        self._print(f"\n{model_name} Results:")
        for metric, value in metrics.items():
            self._print(f"  {metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def evaluate_multiple_models(self, models, X_test, y_test):
        """
        Evaluate multiple models and compare results
        
        Parameters:
        -----------
        models : dict
            Dictionary of model name to trained model
        X_test : array-like
            Test features
        y_test : array-like
            Test target
            
        Returns:
        --------
        pd.DataFrame
            Comparison table of all models
        """
        self._print("Evaluating multiple models...")
        
        results = {}
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, name)
            results[name] = metrics
        
        # Create comparison dataframe
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        self._print("\nModel Comparison:")
        print(results_df)
        
        return results_df
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, 
                             title='Confusion Matrix', save_path=None):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        labels : list, optional
            Class labels
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            self._print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_proba, title='ROC Curve', save_path=None):
        """
        Plot ROC curve
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_proba : array-like
            Predicted probabilities
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        if len(np.unique(y_true)) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
                self._print(f"ROC curve saved to {save_path}")
            
            plt.show()
        else:
            self._print("ROC curve plotting only supported for binary classification")
    
    def plot_precision_recall_curve(self, y_true, y_proba, 
                                    title='Precision-Recall Curve', save_path=None):
        """
        Plot precision-recall curve
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_proba : array-like
            Predicted probabilities
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            self._print(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, top_n=20,
                               title='Feature Importance', save_path=None):
        """
        Plot feature importance
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model with feature_importances_ attribute
        feature_names : list
            List of feature names
        top_n : int
            Number of top features to display
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        if not hasattr(model, 'feature_importances_'):
            self._print("Model does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            self._print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curve(self, model, X, y, cv=5, 
                          title='Learning Curve', save_path=None):
        """
        Plot learning curve
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        X : array-like
            Features
        y : array-like
            Target
        cv : int
            Number of cross-validation folds
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=config.RANDOM_STATE
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 8))
        plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, label='Validation score', color='red', marker='s')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='red')
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            self._print(f"Learning curve saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results_df, metric='accuracy',
                            title='Model Comparison', save_path=None):
        """
        Plot bar chart comparing different models
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with model results
        metric : str
            Metric to compare
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        plt.figure(figsize=(12, 8))
        results_df[metric].sort_values().plot(kind='barh', color='skyblue')
        plt.xlabel(metric.capitalize())
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            self._print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred, target_names=None):
        """
        Generate and print classification report
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        target_names : list, optional
            Target class names
            
        Returns:
        --------
        str
            Classification report
        """
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("\nClassification Report:")
        print(report)
        return report
    
    def save_results(self, results_df, filename='model_results.csv'):
        """
        Save evaluation results to CSV
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results dataframe
        filename : str
            Output filename
        """
        filepath = config.REPORTS_DIR / filename
        results_df.to_csv(filepath)
        self._print(f"Results saved to {filepath}")


def create_evaluation_report(models, X_test, y_test, output_dir=None):
    """
    Create comprehensive evaluation report with all plots
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    output_dir : str, optional
        Directory to save plots
        
    Returns:
    --------
    pd.DataFrame
        Comparison results
    """
    if output_dir is None:
        output_dir = config.FIGURES_DIR
    
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    results_df = evaluator.evaluate_multiple_models(models, X_test, y_test)
    
    # Plot comparison
    evaluator.plot_model_comparison(
        results_df,
        save_path=output_dir / 'model_comparison.png'
    )
    
    # Save results
    evaluator.save_results(results_df)
    
    return results_df