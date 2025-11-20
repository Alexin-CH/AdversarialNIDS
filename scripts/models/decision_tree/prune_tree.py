import numpy as np
from sklearn.tree import DecisionTreeClassifier

def prune_tree_model(dt_model, X_val, y_val, logger=None):
    """
    Perform cost complexity pruning on decision tree.
    
    Parameters:
    -----------
    dt_model : DecisionTreeClassifier
        Trained Decision Tree model.
    X_val : array-like
        Validation features.
    y_val : array-like
        Validation labels.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    pruned_model : DecisionTreeClassifier
        Pruned Decision Tree model.
    best_alpha : float
        Best alpha value found.
    """

    
    # Get cost complexity pruning path
    path = dt_model.cost_complexity_pruning_path(X_val, y_val)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    # Train trees with different alpha values
    scores = []
    for ccp_alpha in ccp_alphas:
        pruned_dt = DecisionTreeClassifier(
            random_state=0,
            ccp_alpha=ccp_alpha,
            class_weight='balanced'
        )
        pruned_dt.fit(X_val, y_val)
        scores.append(pruned_dt.score(X_val, y_val))
    
    # Find best alpha
    best_idx = np.argmax(scores)
    best_alpha = ccp_alphas[best_idx]
    
    # Train final pruned model
    pruned_model = DecisionTreeClassifier(
        random_state=0,
        ccp_alpha=best_alpha,
        class_weight='balanced'
    )
    pruned_model.fit(X_val, y_val)
    
    if logger:
        logger.info(f"Best alpha: {best_alpha:.6f}")
        logger.info(f"Validation score: {scores[best_idx]:.4f}")
    
    return pruned_model, best_alpha