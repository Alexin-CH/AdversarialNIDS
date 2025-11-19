from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def tune_knn_hyperparameters(X_train, y_train, cv=5, n_jobs=-1, logger=None):
    """
    Perform grid search to find optimal KNN hyperparameters.
    
    Parameters:
    -----------
    X_train : array-like
        Training features (should be scaled).
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
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]
    }
    
    if logger:
        logger.info("Starting KNN grid search...")
        logger.info(f"Parameter grid: {param_grid}")
    
    # Create KNN classifier
    knn = KNeighborsClassifier(n_jobs=-1)
    
    # Perform grid search
    grid_search = GridSearchCV(
        knn,
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