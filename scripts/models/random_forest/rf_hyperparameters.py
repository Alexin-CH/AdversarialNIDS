from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
def tune_rf_hyperparameters(X_train, y_train, cv=5, n_jobs=-1, logger=None):
    """
    Perform grid search to find optimal Random Forest hyperparameters.
    
    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    cv : int, default=5
        Number of cross-validation folds.
    n_jobs : int, default=-1
        Number of parallel jobs.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    best_params : dict
        Best hyperparameters found.
    best_score : float
        Best CV score achieved.
    grid_search : GridSearchCV
        Fitted GridSearchCV object.
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    if logger:
        logger.info("Starting Random Forest grid search...")
        logger.info(f"Parameter grid: {param_grid}")
    
    # Create Random Forest classifier
    rf = RandomForestClassifier(random_state=0, class_weight='balanced', n_jobs=-1)
    
    # Perform grid search
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring='accuracy',
        verbose=1 if logger else 0
    )
    
    grid_search.fit(X_train, y_train)
    
    if logger:
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search.best_score_, grid_search