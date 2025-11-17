from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scripts.logger import LoggerManager

def train_random_forest(
    X_train,
    y_train,
    n_estimators=10,
    max_depth=6,
    max_features=None,
    random_state=0,
    cv=5,
    logger=None
):
    """
    Train and evaluate a Random Forest classifier with cross-validation.

    Parameters:
    - X_train: Feature matrix for training.
    - y_train: Target vector for training.
    - n_estimators: Number of trees in the forest (default: 10).
    - max_depth: Maximum depth of each tree (default: 6).
    - max_features: Number of features to consider when looking for the best split (default: None, which means all features).
    - random_state: Controls the randomness for reproducibility (default: 0).
    - cv: Number of folds for cross-validation (default: 5).
    - logger: Optional logger instance for logging messages. If None, uses print.

    Returns:
    - The trained RandomForestClassifier instance.
    - The array of cross-validation scores.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv)
    if logger is not None:
        logger.info('Random Forest Model')
        logger.info('\nCross-validation scores: %s', ', '.join(map(str, cv_scores)))
        logger.info('\nMean cross-validation score: %.2f', cv_scores.mean())
    return rf, cv_scores