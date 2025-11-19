from sklearn.ensemble import RandomForestClassifier

def get_rf_oob_score(X_train, y_train, n_estimators=100, max_depth=10, 
                     random_state=0, logger=None):
    """
    Train Random Forest with out-of-bag scoring.
    
    OOB score is a built-in cross-validation for Random Forest.
    
    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    n_estimators : int, default=100
        Number of trees.
    max_depth : int, default=10
        Maximum depth.
    random_state : int, default=0
        Random state.
    logger : Logger or None
        Optional logger.
    
    Returns:
    --------
    rf : RandomForestClassifier
        Trained model with OOB score.
    oob_score : float
        Out-of-bag score.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        oob_score=True,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    if logger:
        logger.info(f"Random Forest OOB Score: {rf.oob_score_:.4f}")
    
    return rf, rf.oob_score_