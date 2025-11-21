import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

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

def get_tree_feature_importance(model, feature_names, top_n=10, logger=None):
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