from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def train_knn(
    X_train,
    y_train,
    n_neighbors=5,
    weights='uniform',
    metric='minkowski',
    p=2,
    cv=5,
    random_state=0,
    logger=None
):
    """
    Train and evaluate a K-Nearest Neighbors classifier with cross-validation.

    Parameters:
    -----------
    X_train : array-like
        Feature matrix for training.
    y_train : array-like
        Target vector for training.
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : str, default='uniform'
        Weight function used in prediction. Options: 'uniform', 'distance'.
    metric : str, default='minkowski'
        Distance metric to use.
    p : int, default=2
        Power parameter for the Minkowski metric (p=2 is Euclidean).
    cv : int, default=5
        Number of folds for cross-validation.
    random_state : int, default=0
        Random state for reproducibility.
    logger : Logger or None, default=None
        Optional logger instance for logging messages.

    Returns:
    --------
    knn : KNeighborsClassifier
        The trained KNeighborsClassifier instance.
    cv_scores : ndarray
        Array of cross-validation scores.
    """
    # Initialize KNN
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p,
        n_jobs=-1
    )
    
    # Perform cross-validation
    if logger is not None:
        logger.info("Performing KNN cross-validation...")
    
    cv_scores = cross_val_score(knn, X_train, y_train, cv=cv, n_jobs=-1)
    
    # Train on full training set
    if logger is not None:
        logger.info("Training KNN on full training set...")
    
    knn.fit(X_train, y_train)
    
    # Log results
    if logger is not None:
        logger.info('=' * 50)
        logger.info('K-NEAREST NEIGHBORS MODEL')
        logger.info('=' * 50)
        logger.info(f'Parameters: n_neighbors={n_neighbors}, weights={weights}, '
                   f'metric={metric}, p={p}')
        logger.info(f'Cross-validation scores: {cv_scores}')
        logger.info(f'Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
    
    return knn, cv_scores