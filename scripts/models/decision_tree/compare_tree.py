from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
def compare_tree_depths(X_train, y_train, depth_range=range(5, 31, 5), 
                        cv=5, logger=None):
    """
    Compare Decision Tree performance across different max_depth values.
    
    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    depth_range : iterable, default=range(5, 31, 5)
        Range of depth values to test.
    cv : int, default=5
        Number of cross-validation folds.
    logger : Logger or None
        Optional logger.
    
    Returns:
    --------
    results : dict
        Dictionary with depths, scores, and optimal depth.
    """
    depths = list(depth_range)
    mean_scores = []
    std_scores = []
    
    if logger:
        logger.info(f"Testing tree depths: {depths}")
    
    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=0, class_weight='balanced')
        scores = cross_val_score(dt, X_train, y_train, cv=cv, n_jobs=-1)
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
        
        if logger:
            logger.info(f"depth={depth}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Find optimal depth
    optimal_idx = np.argmax(mean_scores)
    optimal_depth = depths[optimal_idx]
    optimal_score = mean_scores[optimal_idx]
    
    if logger:
        logger.info(f"\n✓ Optimal depth: {optimal_depth} with CV score: {optimal_score:.4f}")
    
    return {
        'depths': depths,
        'mean_scores': mean_scores,
        'std_scores': std_scores,
        'optimal_depth': optimal_depth,
        'optimal_score': optimal_score
    }