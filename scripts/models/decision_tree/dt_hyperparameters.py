from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def tune_dt_hyperparameters(X_train, y_train, cv=5, n_jobs=-1, logger=None):
    """
    Perform grid search to find optimal Decision Tree hyperparameters.
    """
    # Define parameter grid
    param_grid = {
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy'],
        'max_features': [None, 'sqrt', 'log2']
    }
    
    if logger:
        logger.info("Starting Decision Tree grid search...")
        logger.info(f"Parameter grid: {param_grid}")
    
    # Create Decision Tree classifier
    dt = DecisionTreeClassifier(random_state=0, class_weight='balanced')
    
    # Perform grid search
    grid_search = GridSearchCV(
        dt,
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