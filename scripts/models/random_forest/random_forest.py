from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

def train_random_forest(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=0,
    cv=5,
    class_weight='balanced',
    logger=None
):
    """
    Train and evaluate a Random Forest classifier with cross-validation.

    Parameters:
    -----------
    X_train : array-like
        Feature matrix for training.
    y_train : array-like
        Target vector for training.
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=10
        Maximum depth of each tree.
    min_samples_split : int, default=5
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=2
        Minimum samples required to be at a leaf node.
    max_features : str, int, float or None, default='sqrt'
        Number of features to consider when looking for the best split.
    random_state : int, default=0
        Controls the randomness for reproducibility.
    cv : int, default=5
        Number of folds for cross-validation.
    class_weight : str or dict, default='balanced'
        Weights associated with classes.
    logger : Logger or None, default=None
        Optional logger instance for logging messages.

    Returns:
    --------
    rf : RandomForestClassifier
        The trained RandomForestClassifier instance.
    cv_scores : ndarray
        Array of cross-validation scores.
    """
    # Initialize Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Perform cross-validation
    if logger is not None:
        logger.info("Performing Random Forest cross-validation...")
    
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, n_jobs=-1)
    
    # Train the model on full training set
    if logger is not None:
        logger.info("Training Random Forest on full training set...")
    
    rf.fit(X_train, y_train)
    
    # Log results
    if logger is not None:
        logger.info('=' * 50)
        logger.info('RANDOM FOREST MODEL')
        logger.info('=' * 50)
        logger.info(f'Parameters: n_estimators={n_estimators}, max_depth={max_depth}, '
                   f'max_features={max_features}')
        logger.info(f'Cross-validation scores: {cv_scores}')
        logger.info(f'Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
    
    return rf, cv_scores