from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

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
    cv_test=False,
    cv=5,
    logger=None
):
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
    if cv_test:
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
    if cv_test:
        logger.info(f'Cross-validation scores: {cv_scores}')
        logger.info(f'Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
        return dt, cv_scores
    else:
        return dt, None





