"""
Decision Tree Classifier Module

Provides functions specific to Decision Tree training and configuration.
"""

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def create_dt_pipeline(
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='gini',
    max_features=None,
    class_weight='balanced',
    random_state=0,
    use_smote=True,
    use_scaler=False
):
    """
    Create a Decision Tree pipeline with preprocessing steps.
    
    Note: Decision trees don't require feature scaling.
    
    Parameters:
    -----------
    max_depth : int, default=15
        Maximum depth of the tree.
    min_samples_split : int, default=10
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=5
        Minimum samples required to be at a leaf node.
    criterion : str, default='gini'
        Split quality measure ('gini' or 'entropy').
    max_features : int, float, str or None, default=None
        Number of features to consider for best split.
    class_weight : str or dict, default='balanced'
        Class weights.
    random_state : int, default=0
        Random state for reproducibility.
    use_smote : bool, default=True
        Whether to apply SMOTE oversampling.
    use_scaler : bool, default=False
        Whether to apply StandardScaler (not needed for DT).
    
    Returns:
    --------
    pipeline : ImbPipeline
        Complete preprocessing and training pipeline.
    """
    steps = []
    
    # Optional scaling (not typically needed for decision trees)
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    
    # Add SMOTE
    if use_smote:
        steps.append(('smote', SMOTE(sampling_strategy='auto', random_state=random_state)))
    
    # Add Decision Tree
    steps.append(('dt', DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state
    )))
    
    return ImbPipeline(steps)


def train_decision_tree(
    X_train,
    y_train,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='gini',
    max_features=None,
    class_weight='balanced',
    random_state=0,
    cv=5,
    logger=None
):
    """
    Train and evaluate a Decision Tree classifier with cross-validation.

    Parameters:
    -----------
    X_train : array-like
        Feature matrix for training.
    y_train : array-like
        Target vector for training.
    max_depth : int, default=15
        Maximum depth of the tree.
    min_samples_split : int, default=10
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=5
        Minimum samples required to be at a leaf node.
    criterion : str, default='gini'
        Function to measure split quality ('gini' or 'entropy').
    max_features : int, float, str or None, default=None
        Number of features to consider when looking for the best split.
    class_weight : str or dict, default='balanced'
        Weights associated with classes.
    random_state : int, default=0
        Controls the randomness for reproducibility.
    cv : int, default=5
        Number of folds for cross-validation.
    logger : Logger or None, default=None
        Optional logger instance for logging messages.

    Returns:
    --------
    dt : DecisionTreeClassifier
        The trained DecisionTreeClassifier instance.
    cv_scores : ndarray
        Array of cross-validation scores.
    """
    # Initialize Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state
    )
    
    # Perform cross-validation
    if logger is not None:
        logger.info("Performing Decision Tree cross-validation...")
    
    cv_scores = cross_val_score(dt, X_train, y_train, cv=cv, n_jobs=-1)
    
    # Train on full training set
    if logger is not None:
        logger.info("Training Decision Tree on full training set...")
    
    dt.fit(X_train, y_train)
    
    # Log results
    if logger is not None:
        logger.info('=' * 50)
        logger.info('DECISION TREE MODEL')
        logger.info('=' * 50)
        logger.info(f'Parameters: max_depth={max_depth}, criterion={criterion}, '
                   f'min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}')
        logger.info(f'Cross-validation scores: {cv_scores}')
        logger.info(f'Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
    
    return dt, cv_scores


def analyze_tree_complexity(model, logger=None):
    """
    Analyze decision tree complexity metrics to detect overfitting.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier or Pipeline
        Trained Decision Tree model or pipeline.
    logger : Logger or None
        Optional logger instance.
    
    Returns:
    --------
    complexity : dict
        Dictionary containing tree complexity metrics.
    """
    # Extract DT model if it's in a pipeline
    if hasattr(model, 'named_steps'):
        dt_model = model.named_steps['dt']
    else:
        dt_model = model
    
    tree = dt_model.tree_
    
    complexity = {
        'n_nodes': tree.node_count,
        'n_leaves': tree.n_leaves,
        'max_depth': tree.max_depth,
        'n_features_used': len(np.unique(tree.feature[tree.feature >= 0]))
    }
    
    if logger is not None:
        logger.info("=" * 50)
        logger.info("TREE COMPLEXITY ANALYSIS")
        logger.info("=" * 50)
        logger.info(f"Total nodes: {complexity['n_nodes']}")
        logger.info(f"Leaf nodes: {complexity['n_leaves']}")
        logger.info(f"Internal nodes: {complexity['n_nodes'] - complexity['n_leaves']}")
        logger.info(f"Max depth: {complexity['max_depth']}")
        logger.info(f"Features used: {complexity['n_features_used']}")
        
        # Warning for very complex trees
        if complexity['n_nodes'] > 1000:
            logger.warning("⚠️  Very large tree detected (>1000 nodes). "
                         "Consider pruning or using Random Forest.")
        if complexity['max_depth'] > 25:
            logger.warning("⚠️  Very deep tree detected (>25 levels). "
                         "May indicate overfitting.")
    
    return complexity


def get_tree_rules(model, feature_names, max_depth=None):
    """
    Extract decision rules from the tree in text format.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier or Pipeline
        Trained Decision Tree model or pipeline.
    feature_names : list
        List of feature names.
    max_depth : int or None
        Maximum depth to display. None for full tree.
    
    Returns:
    --------
    rules : str
        Text representation of decision rules.
    """
    # Extract DT model if it's in a pipeline
    if hasattr(model, 'named_steps'):
        dt_model = model.named_steps['dt']
    else:
        dt_model = model
    
    rules = export_text(dt_model, feature_names=feature_names, max_depth=max_depth)
    return rules


def tune_dt_hyperparameters(X_train, y_train, cv=5, n_jobs=-1, logger=None):
    """
    Perform grid search to find optimal Decision Tree hyperparameters.
    
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


def prune_tree(model, X_val, y_val, logger=None):
    """
    Perform cost complexity pruning on decision tree.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
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
    # Extract DT model if it's in a pipeline
    if hasattr(model, 'named_steps'):
        dt_model = model.named_steps['dt']
    else:
        dt_model = model
    
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