"""
K-Nearest Neighbors Classifier Module

Provides functions specific to KNN training and configuration.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def create_knn_pipeline(
    n_neighbors=5,
    weights='distance',
    metric='minkowski',
    p=2,
    random_state=0,
    use_smote=True,
    use_scaler=True
):
    """
    Create a KNN pipeline with preprocessing steps.
    
    Note: KNN REQUIRES feature scaling for good performance.
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : str, default='distance'
        Weight function ('uniform' or 'distance').
    metric : str, default='minkowski'
        Distance metric to use.
    p : int, default=2
        Power parameter for Minkowski metric (2 = Euclidean).
    random_state : int, default=0
        Random state for reproducibility.
    use_smote : bool, default=True
        Whether to apply SMOTE oversampling.
    use_scaler : bool, default=True
        Whether to apply StandardScaler (HIGHLY RECOMMENDED for KNN).
    
    Returns:
    --------
    pipeline : ImbPipeline
        Complete preprocessing and training pipeline.
    """
    steps = []
    
    # KNN REQUIRES scaling for good performance
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    
    # Add SMOTE
    if use_smote:
        steps.append(('smote', SMOTE(sampling_strategy='auto', random_state=random_state)))
    
    # Add KNN
    steps.append(('knn', KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p,
        n_jobs=-1
    )))
    
    return ImbPipeline(steps)


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


def find_optimal_k(X_train, y_train, k_range=range(3, 31, 2), cv=5, logger=None):
    """
    Find the optimal number of neighbors using cross-validation.
    
    Parameters:
    -----------
    X_train : array-like
        Training features (should be scaled).
    y_train : array-like
        Training labels.
    k_range : iterable, default=range(3, 31, 2)
        Range of k values to test.
    cv : int, default=5
        Number of cross-validation folds.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'k_values': list of k values tested
        - 'mean_scores': list of mean CV scores
        - 'std_scores': list of std CV scores
        - 'optimal_k': best k value
        - 'optimal_score': best CV score
    """
    k_values = list(k_range)
    mean_scores = []
    std_scores = []
    
    if logger:
        logger.info(f"Testing k values: {k_values}")
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, n_jobs=-1)
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
        
        if logger:
            logger.info(f"k={k}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Find optimal k
    optimal_idx = np.argmax(mean_scores)
    optimal_k = k_values[optimal_idx]
    optimal_score = mean_scores[optimal_idx]
    
    if logger:
        logger.info(f"\n✓ Optimal k: {optimal_k} with CV score: {optimal_score:.4f}")
    
    return {
        'k_values': k_values,
        'mean_scores': mean_scores,
        'std_scores': std_scores,
        'optimal_k': optimal_k,
        'optimal_score': optimal_score
    }


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


def estimate_knn_memory(n_samples, n_features, logger=None):
    """
    Estimate memory usage for KNN model.
    
    KNN stores all training data, so memory can be significant.
    
    Parameters:
    -----------
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features.
    logger : Logger or None
        Optional logger.
    
    Returns:
    --------
    memory_mb : float
        Estimated memory in megabytes.
    """
    # Assuming float64 (8 bytes per value)
    bytes_per_value = 8
    total_bytes = n_samples * n_features * bytes_per_value
    memory_mb = total_bytes / (1024 ** 2)
    
    if logger:
        logger.info(f"Estimated KNN memory usage: {memory_mb:.2f} MB")
        logger.info(f"  ({n_samples} samples × {n_features} features × {bytes_per_value} bytes)")
    
    return memory_mb