"""
Model Utilities Module

Shared utility functions for data preprocessing, evaluation, and diagnostics
that are reused across all machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold


def evaluate_model(model, X_test, y_test, logger=None):
    """
    Evaluate trained model on test set with comprehensive metrics.
    
    Parameters:
    -----------
    model : estimator
        Trained classifier model or pipeline.
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    logger : Logger or None, default=None
        Optional logger instance.
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'accuracy': float, test set accuracy
        - 'predictions': array, predicted labels
        - 'report': str, classification report
        - 'confusion_matrix': array, confusion matrix
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Log results
    if logger is not None:
        logger.info(f'Test set accuracy: {acc:.4f}')
        logger.info(f'Classification report:\n{report}')
    
    return {
        'accuracy': acc,
        'predictions': y_pred,
        'report': report,
        'confusion_matrix': cm
    }


def check_data_leakage(X, y, logger=None):
    """
    Perform diagnostic checks for potential data leakage.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix.
    y : Series
        Target labels.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    diagnostics : dict
        Dictionary containing diagnostic information:
        - 'duplicates': int, number of duplicate rows
        - 'duplicate_pct': float, percentage of duplicates
        - 'high_correlation_features': dict, features highly correlated with target
        - 'class_distribution': dict, count of each class
        - 'class_balance_ratio': float, min/max class ratio
        - 'constant_features': list, features with only one value
    """
    diagnostics = {}
    
    # Check for duplicate rows
    duplicates = X.duplicated().sum()
    diagnostics['duplicates'] = duplicates
    diagnostics['duplicate_pct'] = duplicates / len(X) * 100
    
    # Check for features with perfect correlation to target
    if isinstance(y.iloc[0], (int, float)):
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        perfect_corr = correlations[correlations > 0.99]
        diagnostics['high_correlation_features'] = perfect_corr.to_dict()
    else:
        diagnostics['high_correlation_features'] = {}
    
    # Check class distribution
    class_counts = y.value_counts()
    diagnostics['class_distribution'] = class_counts.to_dict()
    diagnostics['class_balance_ratio'] = class_counts.min() / class_counts.max()
    
    # Check for constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    diagnostics['constant_features'] = constant_features
    
    # Log results
    if logger is not None:
        logger.info("=" * 50)
        logger.info("DATA LEAKAGE DIAGNOSTICS")
        logger.info("=" * 50)
        logger.info(f"Duplicate rows: {duplicates} ({diagnostics['duplicate_pct']:.2f}%)")
        
        if diagnostics['high_correlation_features']:
            logger.warning(f"Features with >0.99 correlation to target: "
                         f"{diagnostics['high_correlation_features']}")
        else:
            logger.info("No features with suspiciously high correlation to target")
        
        logger.info(f"Class distribution:\n{class_counts}")
        logger.info(f"Class balance ratio: {diagnostics['class_balance_ratio']:.4f}")
        
        if constant_features:
            logger.warning(f"Constant features found: {constant_features}")
        else:
            logger.info("No constant features found")
    
    return diagnostics


def remove_low_variance_features(X, threshold=0.01, logger=None):
    """
    Remove features with very low variance.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix.
    threshold : float, default=0.01
        Variance threshold below which features are removed.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    X_filtered : DataFrame
        Feature matrix with low variance features removed.
    removed_features : list
        List of removed feature names.
    """
    variances = X.var()
    low_var_features = variances[variances < threshold].index.tolist()
    
    if logger and len(low_var_features) > 0:
        logger.info(f"Removing {len(low_var_features)} low variance features "
                   f"(threshold={threshold})")
    
    X_filtered = X.drop(columns=low_var_features)
    
    return X_filtered, low_var_features


def prepare_data(data, target_column='Attack Type', leakage_features=None, 
                 remove_low_var=True, var_threshold=0.01, logger=None):
    """
    Comprehensive data preparation pipeline.
    
    Parameters:
    -----------
    data : DataFrame
        Raw dataset.
    target_column : str, default='Attack Type'
        Name of target column.
    leakage_features : list or None
        List of features to remove (data leakage).
    remove_low_var : bool, default=True
        Whether to remove low variance features.
    var_threshold : float, default=0.01
        Variance threshold for feature removal.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    X : DataFrame
        Prepared feature matrix.
    y : Series
        Target labels.
    removed_features : dict
        Dictionary with 'leakage' and 'low_variance' feature lists.
    """
    removed_features = {'leakage': [], 'low_variance': []}
    
    # Split features and labels
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Remove leakage features
    if leakage_features is not None:
        existing_leakage = [f for f in leakage_features if f in X.columns]
        if existing_leakage:
            if logger:
                logger.warning(f"🚨 REMOVING LEAKAGE FEATURES: {existing_leakage}")
            X = X.drop(columns=existing_leakage)
            removed_features['leakage'] = existing_leakage
    
    # Convert to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        n_missing = X.isnull().sum().sum()
        if logger:
            logger.info(f"Filling {n_missing} missing values with 0")
        X = X.fillna(0)
    
    # Remove low variance features
    if remove_low_var:
        X, low_var = remove_low_variance_features(X, threshold=var_threshold, logger=logger)
        removed_features['low_variance'] = low_var
    
    if logger:
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, removed_features


def get_feature_importance(model, feature_names, top_n=10, logger=None):
    """
    Extract feature importance from tree-based models.
    
    Works with: RandomForest, DecisionTree, GradientBoosting, XGBoost.
    
    Parameters:
    -----------
    model : estimator or Pipeline
        Trained model or pipeline containing a model.
    feature_names : list
        List of feature names.
    top_n : int, default=10
        Number of top features to return.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    top_features : list of tuples
        List of (feature_name, importance) tuples sorted by importance.
    """
    # Extract model from pipeline if needed
    if hasattr(model, 'named_steps'):
        # Try common model names
        for name in ['rf', 'dt', 'gb', 'xgb', 'model']:
            if name in model.named_steps:
                base_model = model.named_steps[name]
                break
        else:
            raise ValueError("Could not find model in pipeline")
    else:
        base_model = model
    
    # Check if model has feature_importances_
    if not hasattr(base_model, 'feature_importances_'):
        if logger:
            logger.warning("Model does not have feature_importances_ attribute")
        return []
    
    # Get feature importances
    importances = base_model.feature_importances_
    
    # Create list of (feature, importance) tuples
    feature_importance = list(zip(feature_names, importances))
    
    # Sort by importance (descending)
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N features
    top_features = feature_importance[:top_n]
    
    # Log results
    if logger is not None:
        logger.info(f'Top {top_n} most important features:')
        for i, (feature, importance) in enumerate(top_features, 1):
            logger.info(f'{i}. {feature}: {importance:.4f}')
    
    return top_features


def standardize_features(X_train, X_test=None):
    """
    Standardize features using StandardScaler.
    
    Parameters:
    -----------
    X_train : array-like
        Training features.
    X_test : array-like or None
        Test features. If None, only transforms training data.
    
    Returns:
    --------
    X_train_scaled : array
        Scaled training features.
    X_test_scaled : array or None
        Scaled test features (if X_test provided).
    scaler : StandardScaler
        Fitted scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, None, scaler


def select_k_best_features(X_train, y_train, X_test=None, k=30):
    """
    Select k best features using ANOVA F-value.
    
    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    X_test : array-like or None
        Test features.
    k : int, default=30
        Number of features to select.
    
    Returns:
    --------
    X_train_selected : array
        Training features with k best features.
    X_test_selected : array or None
        Test features with k best features (if X_test provided).
    selector : SelectKBest
        Fitted selector object.
    selected_indices : array
        Indices of selected features.
    """
    selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    selected_indices = selector.get_support(indices=True)
    
    if X_test is not None:
        X_test_selected = selector.transform(X_test)
        return X_train_selected, X_test_selected, selector, selected_indices
    
    return X_train_selected, None, selector, selected_indices


def balance_classes_info(y, logger=None):
    """
    Display class balance information.
    
    Parameters:
    -----------
    y : array-like
        Target labels.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    info : dict
        Dictionary with class balance information.
    """
    class_counts = pd.Series(y).value_counts()
    total = len(y)
    
    info = {
        'class_counts': class_counts.to_dict(),
        'class_percentages': (class_counts / total * 100).to_dict(),
        'n_classes': len(class_counts),
        'imbalance_ratio': class_counts.min() / class_counts.max(),
        'majority_class': class_counts.idxmax(),
        'minority_class': class_counts.idxmin()
    }
    
    if logger:
        logger.info("=" * 50)
        logger.info("CLASS BALANCE INFORMATION")
        logger.info("=" * 50)
        logger.info(f"Number of classes: {info['n_classes']}")
        logger.info(f"Total samples: {total}")
        logger.info(f"Imbalance ratio: {info['imbalance_ratio']:.4f}")
        logger.info(f"Majority class: {info['majority_class']} ({class_counts.max()} samples)")
        logger.info(f"Minority class: {info['minority_class']} ({class_counts.min()} samples)")
        logger.info("\nClass distribution:")
        for cls, count in class_counts.items():
            pct = count / total * 100
            logger.info(f"  {cls}: {count} ({pct:.2f}%)")
    
    return info


def remove_rare_classes(X, y, min_samples=2, logger=None):
    """
    Remove classes with fewer than min_samples samples.
    
    Required for stratified train-test split.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix.
    y : array-like
        Target labels.
    min_samples : int, default=2
        Minimum number of samples required per class.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    X_filtered : array-like
        Feature matrix with rare classes removed.
    y_filtered : array-like
        Target labels with rare classes removed.
    removed_classes : list
        List of removed class labels.
    """
    class_counts = pd.Series(y).value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    removed_classes = class_counts[class_counts < min_samples].index.tolist()
    
    mask = pd.Series(y).isin(valid_classes)
    
    if isinstance(X, pd.DataFrame):
        X_filtered = X[mask]
    else:
        X_filtered = X[mask.values]
    
    if isinstance(y, pd.Series):
        y_filtered = y[mask]
    else:
        y_filtered = y[mask.values]
    
    if logger and len(removed_classes) > 0:
        logger.warning(f"Removed {len(removed_classes)} rare classes with <{min_samples} samples: "
                      f"{removed_classes}")
    
    return X_filtered, y_filtered, removed_classes


def print_performance_summary(cv_scores, test_accuracy, model_name, 
                             config=None, logger=None):
    """
    Print a formatted performance summary.
    
    Parameters:
    -----------
    cv_scores : array
        Cross-validation scores.
    test_accuracy : float
        Test set accuracy.
    model_name : str
        Name of the model.
    config : dict or None
        Model configuration parameters.
    logger : Logger or None
        Optional logger instance.
    """
    summary = []
    summary.append("=" * 70)
    summary.append(f"{model_name.upper()} PERFORMANCE SUMMARY")
    summary.append("=" * 70)
    summary.append(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    summary.append(f"Test Accuracy: {test_accuracy:.4f}")
    summary.append(f"CV-Test Gap: {abs(cv_scores.mean() - test_accuracy):.4f}")
    
    if config:
        summary.append("\nModel Configuration:")
        for key, value in config.items():
            summary.append(f"  - {key}: {value}")
    
    # Performance interpretation
    summary.append("\nPerformance Assessment:")
    if cv_scores.mean() > 0.99:
        summary.append("  ⚠️  WARNING: CV score > 0.99 may indicate data leakage!")
    elif cv_scores.mean() >= 0.95:
        summary.append("  ✓ Excellent performance (CV ≥ 0.95)")
    elif cv_scores.mean() >= 0.90:
        summary.append("  ✓ Good performance (CV ≥ 0.90)")
    else:
        summary.append("  ⚠️  Performance below 0.90 - consider improvements")
    
    # Overfitting check
    gap = abs(cv_scores.mean() - test_accuracy)
    if gap > 0.05:
        summary.append(f"\n  ⚠️  Large CV-Test gap ({gap:.4f}) suggests overfitting")
    else:
        summary.append(f"\n  ✓ Good generalization (CV-Test gap: {gap:.4f})")
    
    summary_text = "\n".join(summary)
    
    if logger:
        logger.info(summary_text)
    else:
        print(summary_text)