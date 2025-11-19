from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
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