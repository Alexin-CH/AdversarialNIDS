from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

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





